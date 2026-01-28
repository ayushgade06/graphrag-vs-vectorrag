from collections import defaultdict


def _get(r, k):
    return float(r.get(k, 0.0))


def aggregate_results(results):
    grouped = defaultdict(list)

    for r in results:
        grouped[r["dataset"]].append(r)

    summary = {}

    for dataset, rows in grouped.items():
        n = len(rows) if rows else 1

        summary[dataset] = {
            "vector_f1": round(sum(_get(r, "vector_f1") for r in rows) / n, 4),
            "vector_oracle_f1": round(sum(_get(r, "vector_oracle_f1") for r in rows) / n, 4),
            "graph_f1": round(sum(_get(r, "graph_f1") for r in rows) / n, 4),
            "graph_oracle_f1": round(sum(_get(r, "graph_oracle_f1") for r in rows) / n, 4),

            "vector_rouge": round(sum(_get(r, "vector_rouge") for r in rows) / n, 4),
            "vector_oracle_rouge": round(sum(_get(r, "vector_oracle_rouge") for r in rows) / n, 4),
            "graph_rouge": round(sum(_get(r, "graph_rouge") for r in rows) / n, 4),
            "graph_oracle_rouge": round(sum(_get(r, "graph_oracle_rouge") for r in rows) / n, 4),

            # NEW
            "vector_recall": round(sum(_get(r, "vector_recall") for r in rows) / n, 4),
            "graph_recall": round(sum(_get(r, "graph_recall") for r in rows) / n, 4),
        }

        summary[dataset]["delta_f1_llm"] = round(
            summary[dataset]["graph_f1"] - summary[dataset]["vector_f1"], 4
        )
        summary[dataset]["delta_rouge_llm"] = round(
            summary[dataset]["graph_rouge"] - summary[dataset]["vector_rouge"], 4
        )

    return summary


def print_results_table(summary):
    header = (
        f"{'Dataset':<15} | {'Vec F1':<7} | {'Gr F1':<7} | "
        f"{'Vec Or F1':<9} | {'Gr Or F1':<9} | "
        f"{'Vec RG':<7} | {'Gr RG':<7} | "
        f"{'Vec Rec':<7} | {'Gr Rec':<7} | "
        f"{'ΔF1':<6} | {'ΔRG':<6}"
    )
    print(header)
    print("-" * len(header))

    for dataset, m in summary.items():
        print(
            f"{dataset:<15} | "
            f"{m['vector_f1']:<7} | {m['graph_f1']:<7} | "
            f"{m['vector_oracle_f1']:<9} | {m['graph_oracle_f1']:<9} | "
            f"{m['vector_rouge']:<7} | {m['graph_rouge']:<7} | "
            f"{m['vector_recall']:<7} | {m['graph_recall']:<7} | "
            f"{m['delta_f1_llm']:<6} | {m['delta_rouge_llm']:<6}"
        )
