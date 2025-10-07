import numpy as np
from tqdm import tqdm

def levenshtein(ref, hyp):
    m = len(ref)
    n = len(hyp)

    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    i, j = m, n
    S, I, D = 0, 0, 0
    while i > 0 and j > 0:
        if ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i][j-1] + 1:
            I += 1
            j -= 1
        else:
            D += 1
            i -= 1
    D += i
    I += j

    return dp[m][n],  S, I, D

def calculate_error_rates(reference_lines, hypothesis_lines):
    # LER
    ler_total, ler_S, ler_I, ler_D = levenshtein(reference_lines, hypothesis_lines)
    ler = ler_total / len(reference_lines) if reference_lines else 0

    # WER
    ref_words = " ".join(reference_lines).split()
    hyp_words = " ".join(hypothesis_lines).split()
    wer_total, wer_S, wer_I, wer_D = levenshtein(ref_words, hyp_words)
    wer = wer_total / len(ref_words) if ref_words else 0

    # CER
    ref_chars = list("".join(reference_lines))
    hyp_chars = list("".join(hypothesis_lines))
    cer_total, cer_S, cer_I, cer_D = levenshtein(ref_chars, hyp_chars)
    cer = cer_total / len(ref_chars) if ref_chars else 0

    return {
        'LER': {'rate': ler, 'S': ler_S, 'I': ler_I, 'D': ler_D},
        'WER': {'rate': wer, 'S': wer_S, 'I': wer_I, 'D': wer_D},
        'CER': {'rate': cer, 'S': cer_S, 'I': cer_I, 'D': cer_D},
    }
