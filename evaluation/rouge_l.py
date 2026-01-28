def lcs_length(x, y):
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[-1][-1]


def compute_rouge_l(predicted: str, ground_truth: str) -> float:
    if not predicted or not ground_truth:
        return 0.0

    pred = predicted.split()
    gt = ground_truth.split()

    if not pred or not gt:
        return 0.0

    lcs = lcs_length(pred, gt)
    precision = lcs / len(pred)
    recall = lcs / len(gt)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
