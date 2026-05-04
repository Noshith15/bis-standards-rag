"""
inference.py — BIS Hybrid RAG Inference Entry Point
Usage: python inference.py --input hidden_private_dataset.json --output team_results.json
"""
import json, argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from hybrid_retriever import HybridBISRetriever

def run(input_path, output_path):
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'standards.json')
    retriever = HybridBISRetriever(data_path)
    queries = json.load(open(input_path))
    results = []
    for item in queries:
        retrieved, latency = retriever.retrieve(item['query'], top_k=5)
        results.append({
            "id": item['id'],
            "retrieved_standards": [r['standard_id'] for r in retrieved],
            "latency_seconds": latency
        })
        print(f"  {item['id']} → {retrieved[0]['standard_id']} ({latency}s)")
    json.dump(results, open(output_path,'w'), indent=2)
    print(f"\n✅ {len(results)} results saved to {output_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(args.input, args.output)
