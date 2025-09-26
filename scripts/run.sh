source /opt/dlami/nvme/thanh-dev/bin/activate 

source /opt/dlami/nvme/thanh-dev/bin/activate && python eval_v1.py --dataset data/dataset.csv --top-k 3

cd /home/ubuntu/thanh-dev/RAG-benchmark && source /opt/dlami/nvme/thanh-dev/bin/activate && PYTHONPATH=. python scripts/test_pipeline_debug.py --dataset example_data/dev_data.jsonl.bz2 --limit 10 --print_samples 2 --use_gemini --gemini_model gemini-2.5-flash --out data/results/gemini_per_sample.jsonl | tail -n +1 | cat