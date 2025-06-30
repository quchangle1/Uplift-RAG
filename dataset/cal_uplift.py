import json
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_metric_gains(naive_data, zero_data):
    zero_dict = {item["id"]: item for item in zero_data}
    
    results = []
    
    for naive_item in naive_data:
        base_id = "_".join(naive_item["id"].split("_")[:2])
        
        zero_item = zero_dict.get(base_id)
            
        delta_metrics = {}
        naive_metrics = naive_item["output"]["metric_score"]
        zero_metrics = zero_item["output"]["metric_score"]
        
        for metric in naive_metrics:
            delta = naive_metrics[metric] - zero_metrics[metric]
            delta_metrics[metric] = delta
        
        result = {
            "doc_id": naive_item["id"],
            "base_id": base_id,
            "question": naive_item["question"],
            "metric_delta": delta_metrics
        }
        results.append(result)
    
    return results



def process_files(score_file, retrieval_file, output_file):
    with open(score_file, "r") as f:
        score_data = json.load(f)
    
    with open(retrieval_file, "r") as f:
        retrieval_data = json.load(f)
    
    query_scores = defaultdict(lambda: {"f1_scores": [], "doc_ids": []})
    for entry in score_data:
        base_id = entry["base_id"]
        f1_score = entry["metric_delta"]["f1"]
        doc_id = entry["doc_id"]
        question = entry["question"]

        query_scores[base_id]["question"] = question
        query_scores[base_id]["f1_scores"].append(f1_score)
        query_scores[base_id]["doc_ids"].append(doc_id)
    
    retrieval_results = defaultdict(list)
    for entry in retrieval_data:
        retrieval_results[entry["id"]] = [doc["contents"] for doc in entry["output"]["retrieval_result"]]
    
    output_data = []
    for query_id, values in query_scores.items():
        f1_scores = values["f1_scores"]
        doc_ids = values["doc_ids"]
        question = values["question"]

        positive_f1_scores = {f1 for f1 in f1_scores if f1 > 0}
        list_label = 1 if len(positive_f1_scores) > 1 else 0
        
        unique_f1_scores = set(f1_scores)
        pair_label = 1 if len(unique_f1_scores) > 1 else 0

        uplift_label = [1 if f1 > 0 else 0 fo
        retrieval_docs = []
        for doc_id in doc_ids:
            retrieval_docs.extend(retrieval_results.get(doc_id, []))
        
        output_data.append({
            "query_id": query_id,
            "question": question,
            "list_label": list_label,
            "pair_label": pair_label,
            "retrieval_docs": retrieval_docs,
            "f1": f1_scores,
            "uplift_label": uplift_label
        })
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)



def main():
    retrieval_file = ""
    naive_data = load_json(retrieval_file)
    zero_data = load_json("")
    output_file = ""
    data_file = ""
    metric_gains = calculate_metric_gains(naive_data, zero_data)
    
    with open(output_file, "w") as f:
        json.dump(metric_gains, f, indent=2)

    process_files(output_file, retrieval_file, data_file)

if __name__ == "__main__":
    main()
