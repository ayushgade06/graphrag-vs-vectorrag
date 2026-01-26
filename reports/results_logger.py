from collections import defaultdict


def aggregate_results(results):
    grouped = defaultdict(list)

    for r in results:
        grouped[r["dataset"]].append(r)

    summary = {}

    for dataset, rows in grouped.items():
        summary[dataset] = {
            "vector_f1": round(sum(r["vector_f1"] for r in rows) / len(rows), 4),
            "vector_rouge": round(sum(r["vector_rouge"] for r in rows) / len(rows), 4),
            "graph_f1": round(sum(r["graph_f1"] for r in rows) / len(rows), 4),
            "graph_rouge": round(sum(r["graph_rouge"] for r in rows) / len(rows), 4),
            "delta_f1": round(
                (sum(r["graph_f1"] for r in rows) - sum(r["vector_f1"] for r in rows)) / len(rows),
                4
            ),
            "delta_rouge": round(
                (sum(r["graph_rouge"] for r in rows) - sum(r["vector_rouge"] for r in rows)) / len(rows),
                4
            ),
        }

    return summary


def print_results_table(summary):
    header = (
        f"{'Dataset':<15} | {'Vec F1':<8} | {'Gr F1':<8} | {'ΔF1':<8} | "
        f"{'Vec RG':<8} | {'Gr RG':<8} | {'ΔRG':<8}"
    )
    print(header)
    print("-" * len(header))

    for dataset, m in summary.items():
        print(
            f"{dataset:<15} | {m['vector_f1']:<8} | {m['graph_f1']:<8} | {m['delta_f1']:<8} | "
            f"{m['vector_rouge']:<8} | {m['graph_rouge']:<8} | {m['delta_rouge']:<8}"
        )
