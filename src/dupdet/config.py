"""Fixed settings of the paper."""

# bootstrap (Sect. 3.6)
SEED = 2026
N_BOOTSTRAPS = 10

# MinHash and banding (Sect. 3.4): b = n // r, only the first b * r values are used
N_HASHES = 500
ROWS_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20)
BR_GRID = [(N_HASHES // r, r) for r in ROWS_VALUES]

# title tokens (Sect. 3.1)
DF_CUTOFFS = (0.1, 0.2, 0.3, 0.5)    # c, chosen by training F1*
MIN_DF = 2

# key-value tokens (Sect. 3.2, Appendix A)
NUMERIC_SHARE = 0.8      # a key is numeric if >= 80% of its values contain a digit
MIN_PAIRS = 10           # |D_kk'| >= 10; also the minimum evidence for a key weight
ALIGN_M_MIN = 0.8        # m_kk' >= 0.8
ALIGN_LIFT_MIN = 1.0     # log(m_kk' / u_kk') >= 1
N_NONDUP_ALIGN = 20000   # sampled non-duplicate pairs for the alignment
N_NONDUP_WEIGHTS = 50000  # sampled non-duplicate pairs for m and u
TOP_K_KEYS = 10

# graded weights (Sect. 3.3): s and s_k, chosen by training F1*
GRADE_SCALES = (0.5, 1, 2)

# classifier (Sect. 3.5)
PROB_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))
MIN_TRAIN_POS = 1        # positive training candidates needed to fit the model

# MSM benchmark: beta and q fixed, full grid for alpha, gamma, mu and epsilon
MSM_Q = 3
MSM_BETA = 0.0
MSM_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
