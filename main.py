#!/usr/bin/env python3
"""
Simple RAG Chat Terminal Application
Add documents and chat with RAG system.
"""

import argparse
import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Ensure project modules are importable when running directly
if 'pipeline' not in sys.path:
    sys.path.insert(0, 'pipeline')
if 'src' not in sys.path:
    sys.path.insert(0, 'src')

# Import for DEFAULT RAG system (retrieval_augmented_generation)
try:
    from retrieval_augmented_generation.documents_manager import DocumentsManager as DocumentsManager_default
    from retrieval_augmented_generation.document_loaders import InputDocument as InputDocument_default, InputSource as InputSource_default
    # Import modules to register with factories
    import retrieval_augmented_generation.vectorstores.qdrant_vectorstore  # noqa: F401
    import retrieval_augmented_generation.embeddings.openai_embeddings  # noqa: F401
    import retrieval_augmented_generation.embeddings.sentence_transformer_embeddings  # noqa: F401
    import retrieval_augmented_generation.rerankings.cross_encoder_reranking  # noqa: F401
    try:
        import retrieval_augmented_generation.rerankings.flag_reranking  # noqa: F401
    except ImportError:
        pass  # Skip FlagEmbedding if not available
    import retrieval_augmented_generation.document_loaders.pdf_processor  # noqa: F401
    import retrieval_augmented_generation.document_loaders.text_processor  # noqa: F401
    import retrieval_augmented_generation.document_loaders.url_processor  # noqa: F401
    DEFAULT_AVAILABLE = True
except ImportError as e:
    DEFAULT_AVAILABLE = False
    print(f"Warning: DEFAULT system not available: {e}")

# Import for V1 RAG system (retrieval_augmented_generation_v1) as fallback
try:
    from retrieval_augmented_generation_v1.documents_manager import DocumentsManager as DocumentsManager_v1
    from retrieval_augmented_generation_v1.document_loaders import InputDocument as InputDocument_v1, InputSource as InputSource_v1
    # Import modules to register with factories
    import retrieval_augmented_generation_v1.vectorstores.qdrant_vectorstore  # noqa: F401
    import retrieval_augmented_generation_v1.embeddings.openai_embeddings  # noqa: F401
    import retrieval_augmented_generation_v1.embeddings.sentence_transformer_embeddings  # noqa: F401
    import retrieval_augmented_generation_v1.rerankings.cross_encoder_reranking  # noqa: F401
    try:
        import retrieval_augmented_generation_v1.rerankings.flag_reranking  # noqa: F401
    except ImportError:
        pass  # Skip FlagEmbedding if not available
    import retrieval_augmented_generation_v1.document_loaders.pdf_processor  # noqa: F401
    import retrieval_augmented_generation_v1.document_loaders.text_processor  # noqa: F401
    import retrieval_augmented_generation_v1.document_loaders.url_processor  # noqa: F401
    V1_AVAILABLE = True
except ImportError as e:
    V1_AVAILABLE = False
    print(f"Warning: V1 system not available: {e}")


class SimpleRAGChatApp:
    """Simple RAG Chat Terminal Application - Add Documents and Chat Only"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize systems
        self.v1_manager = None
        self.default_manager = None
        self.v1_rag_system = None
        self.default_rag_system = None
        self.current_system = "default"  # Default to DEFAULT system
        self.current_collection = None
        
        # Initialize available systems
        self.initialize_systems()
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            sys.exit(1)
    
    def initialize_systems(self):
        """Initialize both RAG systems."""
        # Initialize DEFAULT system
        if DEFAULT_AVAILABLE:
            try:
                self.default_manager = DocumentsManager_default(config=self.config, logger=self.logger)
                from src.rag_system import SimpleRAGSystem
                self.default_rag_system = SimpleRAGSystem(self.default_manager, self.logger)
                self.logger.info("✅ DEFAULT RAG system initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize DEFAULT system: {e}")
        
        # Initialize V1 system
        if V1_AVAILABLE:
            try:
                self.v1_manager = DocumentsManager_v1(config=self.config, logger=self.logger)
                from src.rag_system import SimpleRAGSystem
                self.v1_rag_system = SimpleRAGSystem(self.v1_manager, self.logger)
                self.logger.info("✅ V1 RAG system initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize V1 system: {e}")
        
        # Check if any system is available
        if not self.default_manager and not self.v1_manager:
            print("❌ No RAG systems are available. Please check installation.")
        else:
            # Auto-select first available system
            if self.current_system == "default" and not self.default_manager:
                if self.v1_manager:
                    self.current_system = "v1"
                    print("⚠️  DEFAULT system not available, switched to V1")
            elif self.current_system == "v1" and not self.v1_manager:
                if self.default_manager:
                    self.current_system = "default"
                    print("⚠️  V1 system not available, switched to DEFAULT")
    
    def get_current_manager_and_rag(self):
        """Get the current active document manager and RAG system."""
        if self.current_system == "v1" and self.v1_manager:
            return self.v1_manager, self.v1_rag_system
        elif self.current_system == "default" and self.default_manager:
            return self.default_manager, self.default_rag_system
        else:
            raise RuntimeError(f"System {self.current_system} is not available")
    
    def get_input_classes(self):
        """Get the input document classes for current system."""
        if self.current_system == "v1" and V1_AVAILABLE:
            return InputDocument_v1, InputSource_v1
        elif self.current_system == "default" and DEFAULT_AVAILABLE:
            return InputDocument_default, InputSource_default
        else:
            raise RuntimeError(f"System {self.current_system} is not available")
    
    def print_header(self):
        """Print application header."""
        print("\n" + "="*60)
        print("🤖 SIMPLE RAG CHAT TERMINAL")
        print("="*60)
        print(f"📊 V1 System: {'✅ Available' if self.v1_manager else '❌ Not Available'}")
        print(f"📊 DEFAULT System: {'✅ Available' if self.default_manager else '❌ Not Available'}")
        print(f"🔄 Current System: {self.current_system.upper()}")
        print(f"📁 Current Collection: {self.current_collection or 'None'}")
        print("="*60)
    
    def print_help(self):
        """Print available commands."""
        print("\n📋 Available Commands:")
        print("─" * 40)
        print("🔧 System Management:")
        print("  switch v1/default    - Switch between RAG systems")
        print("")
        print("📁 Collection Management:")
        print("  create <name>        - Create new collection")
        print("  use <name>           - Switch to collection")
        print("")
        print("📄 Document Management:")
        print("  add file <path>      - Add document from file")
        print("  add url <url>        - Add document from URL")  
        print("  add text <content>   - Add text document")
        print("")
        print("💬 RAG Chat:")
        print("  chat <query>         - Chat with RAG (retrieve + generate)")
        print("")
        print("ℹ️  Other:")
        print("  help                 - Show this help")
        print("  clear                - Clear screen")
        print("  exit/quit            - Exit application")
        print("─" * 40)
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    async def switch_system(self, system_name: str):
        """Switch between RAG systems."""
        system_name = system_name.lower()
        
        if system_name == "v1":
            if self.v1_manager:
                self.current_system = "v1"
                print(f"✅ Switched to V1 system")
            else:
                print("❌ V1 system is not available")
        elif system_name == "default":
            if self.default_manager:
                self.current_system = "default"
                print(f"✅ Switched to DEFAULT system")
            else:
                print("❌ DEFAULT system is not available")
        else:
            print(f"❌ Unknown system: {system_name}")
            print("💡 Available systems: v1, default")
    
    async def create_collection(self, collection_name: str):
        """Create a new collection."""
        try:
            document_manager, _ = self.get_current_manager_and_rag()
            
            # Get first available embedding model
            embedding_models = document_manager.get_supported_embedding_models()
            if not embedding_models:
                print("❌ No embedding models available")
                return
            
            model_name = embedding_models[0]
            await document_manager.create_collection(
                collection_name=collection_name,
                model_name=model_name
            )
            self.current_collection = collection_name
            print(f"✅ Collection '{collection_name}' created successfully!")
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
    
    async def add_document_from_file(self, file_path: str):
        """Add document from file."""
        if not self.current_collection:
            print("❌ No collection selected. Use 'create <collection>' first.")
            return
        
        try:
            document_manager, _ = self.get_current_manager_and_rag()
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return
            
            # Create input document with explicit type detection
            InputDocument, InputSource = self.get_input_classes()
            from pathlib import Path
            from retrieval_augmented_generation.document_loaders import DocumentType, InputSource as CorrectInputSource
            
            # Detect document type by extension
            path = Path(file_path)
            ext = path.suffix.lower()
            if ext == ".pdf":
                doc_type = DocumentType.PDF
            elif ext in [".txt", ""]:
                doc_type = DocumentType.TEXT
            else:
                doc_type = DocumentType.TEXT  # Default fallback
            
            input_doc = InputDocument(
                content=file_path,
                source=CorrectInputSource.FILE_PATH,
                type=doc_type,
                metadata={"source": file_path}
            )
            
            loaded_text = await document_manager.add_document(
                collection_name=self.current_collection,
                document=input_doc
            )
            
            print(f"✅ Document added from '{file_path}'")
            print(f"📄 Preview: {loaded_text[:200]}...")
            
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
    
    async def add_document_from_url(self, url: str):
        """Add document from URL."""
        if not self.current_collection:
            print("❌ No collection selected. Use 'create <collection>' first.")
            return
        
        try:
            document_manager, _ = self.get_current_manager_and_rag()
            
            InputDocument, InputSource = self.get_input_classes()
            from retrieval_augmented_generation.document_loaders import DocumentType, InputSource as CorrectInputSource
            
            input_doc = InputDocument(
                content=url,
                source=CorrectInputSource.URL,
                type=DocumentType.URL,
                metadata={"source": url}
            )
            
            loaded_text = await document_manager.add_document(
                collection_name=self.current_collection,
                document=input_doc
            )
            
            print(f"✅ Document added from '{url}'")
            print(f"📄 Preview: {loaded_text[:200]}...")
            
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
    
    async def add_text_document(self, text_content: str):
        """Add text document."""
        if not self.current_collection:
            print("❌ No collection selected. Use 'create <collection>' first.")
            return
        
        try:
            document_manager, _ = self.get_current_manager_and_rag()
            
            InputDocument, InputSource = self.get_input_classes()
            from retrieval_augmented_generation.document_loaders import DocumentType, InputSource as CorrectInputSource
            
            input_doc = InputDocument(
                content=text_content,
                source=CorrectInputSource.RAW_TEXT,
                type=DocumentType.TEXT,
                metadata={"source": "manual_text"}
            )
            
            loaded_text = await document_manager.add_document(
                collection_name=self.current_collection,
                document=input_doc
            )
            
            print(f"✅ Text document added")
            print(f"📄 Preview: {loaded_text[:200]}...")
            
        except Exception as e:
            print(f"❌ Failed to add text document: {e}")
    
    async def chat_with_rag(self, query: str):
        """Chat with RAG: retrieve documents and generate response."""
        if not self.current_collection:
            print("❌ No collection selected. Use 'create <collection>' first.")
            return
        
        try:
            _, rag_system = self.get_current_manager_and_rag()
            
            print("🔄 Processing RAG chat...")
            
            # Use RAG system to get full response
            result = await rag_system.chat_with_rag(
                collection_name=self.current_collection,
                query=query,
                limit=5
            )
            
            # Display formatted result
            formatted_output = rag_system.format_for_display(result)
            print(formatted_output)
            
            return result
            
        except Exception as e:
            print(f"❌ RAG chat failed: {e}")
            import traceback
            print(f"📝 Error details: {traceback.format_exc()}")
            return None
    
    async def process_command(self, user_input: str) -> bool:
        """Process user command. Returns False if should exit."""
        parts = user_input.strip().split()
        if not parts:
            return True
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd in ["exit", "quit"]:
            return False
        
        elif cmd == "help":
            self.print_help()
        
        elif cmd == "clear":
            self.clear_screen()
            self.print_header()
        
        elif cmd == "switch" and args:
            system_name = args[0]
            await self.switch_system(system_name)
        
        elif cmd == "create" and args:
            collection_name = args[0]
            await self.create_collection(collection_name)
        
        elif cmd == "use" and args:
            collection_name = args[0]
            self.current_collection = collection_name
            print(f"✅ Switched to collection: {collection_name}")
        
        elif cmd == "add" and len(args) >= 2:
            add_type = args[0].lower()
            
            if add_type == "file":
                file_path = args[1]
                await self.add_document_from_file(file_path)
            
            elif add_type == "url":
                url = args[1]
                await self.add_document_from_url(url)
            
            elif add_type == "text":
                text_content = " ".join(args[1:])
                await self.add_text_document(text_content)
            
            else:
                print(f"❌ Unknown add type: {add_type}")
                print("💡 Usage: add file <path> | add url <url> | add text <content>")
        
        elif cmd == "chat" and args:
            query = " ".join(args)
            await self.chat_with_rag(query)
        
        else:
            print("❌ Unknown command. Type 'help' for available commands.")
        
        return True
    
    async def run(self):
        """Run the main application loop."""
        self.print_header()
        print("Type 'help' for available commands or 'exit' to quit.")
        
        while True:
            try:
                # Get user input
                prompt = f"[{self.current_system.upper()}:{self.current_collection or 'None'}] > "
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Process command
                should_continue = await self.process_command(user_input)
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                print(f"📝 Error details: {traceback.format_exc()}")
                # Continue to next iteration instead of breaking


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple RAG Chat Terminal Application")
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    app = SimpleRAGChatApp(config_path=args.config)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
