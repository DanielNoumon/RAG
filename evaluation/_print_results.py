"""Pretty-print the latest evaluation results."""
import json
import glob


def main():
    files = sorted(glob.glob("data/eval_results/eval_*.json"))
    if not files:
        print("No eval results found in data/eval_results/")
        return

    path = files[-1]
    print(f"Reading: {path}\n")

    d = json.load(open(path, "r"))
    agg = {
        m: d["methods"][m]["aggregated"] for m in d["methods"]
    }

    keys = [
        "mean_mrr", "map",
        "mean_ndcg@5", "mean_ndcg@10",
        "mean_recall@1", "mean_recall@3",
        "mean_recall@5", "mean_recall@10",
        "mean_latency_s",
    ]

    methods = list(agg.keys())
    header = "{:<22}".format("Metric")
    for m in methods:
        header += "  {:>12}".format(agg[m]["method"])
    print(header)
    print("-" * len(header))

    for k in keys:
        row = "{:<22}".format(k)
        for m in methods:
            row += "  {:>12.4f}".format(agg[m].get(k, 0))
        print(row)


if __name__ == "__main__":
    main()
