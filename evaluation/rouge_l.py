def lcs_length(x, y):
    """
    Compute length of Longest Common Subsequence.
    """
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[-1][-1]


def compute_rouge_l(predicted: str, ground_truth: str) -> float:
    """
    Compute ROUGE-L score.
    """

    pred_tokens = predicted.split()
    gt_tokens = ground_truth.split()

    lcs = lcs_length(pred_tokens, gt_tokens)

    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(gt_tokens) if gt_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
