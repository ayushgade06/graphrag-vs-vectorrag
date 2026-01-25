from collections import defaultdict

def aggregate_results(results):
    """
    Aggregate results dataset-wise.
    """

    aggregated = defaultdict(list)

    for r in results:
        aggregated[r["dataset"]].append(r)

    summary = {}

    for dataset, rows in aggregated.items():
        vector_f1 = sum(r["vector_f1"] for r in rows) / len(rows)
        vector_rouge = sum(r["vector_rouge"] for r in rows) / len(rows)

        graph_f1 = sum(r["graph_f1"] for r in rows) / len(rows)
        graph_rouge = sum(r["graph_rouge"] for r in rows) / len(rows)

        summary[dataset] = {
            "vector_f1": round(vector_f1, 4),
            "vector_rouge": round(vector_rouge, 4),
            "graph_f1": round(graph_f1, 4),
            "graph_rouge": round(graph_rouge, 4),
            "delta_f1": round(graph_f1 - vector_f1, 4),
            "delta_rouge": round(graph_rouge - vector_rouge, 4),
        }

    return summary

def print_results_table(summary):
    """
    Print a formatted table of results.
    """
    header = f"{'Dataset':<15} | {'Vec F1':<8} | {'Gr F1':<8} | {'D-F1':<8} | {'Vec RG':<8} | {'Gr RG':<8} | {'D-RG':<8}"
    print(header)
    print("-" * len(header))

    for dataset, metrics in summary.items():
        print(f"{dataset:<15} | "
              f"{metrics['vector_f1']:<8} | "
              f"{metrics['graph_f1']:<8} | "
              f"{metrics['delta_f1']:<8} | "
              f"{metrics['vector_rouge']:<8} | "
              f"{metrics['graph_rouge']:<8} | "
              f"{metrics['delta_rouge']:<8}")
