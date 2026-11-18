# Scalable Product Duplicate Detection with Learned Key-Value Tokens and Locality-Sensitive Hashing

Borg Knijnenburg — Text Mining and Recommender Systems, Erasmus University Rotterdam.

Repository: https://github.com/Borg-JK/LSH_TextMining-RecommenderSystems

This repository contains the code and results of the paper. Comparing all
pairs of 1,624 televisions from four Web shops takes 1,317,876 comparisons, of
which only 399 are duplicates. Locality-Sensitive Hashing (LSH) selects
candidate pairs; the paper studies which product representation makes LSH
most effective. The title is represented by its informative tokens (a
frequency filter and idf-graded copies), and the key-value pairs by
`key=value` tokens of keys that are aligned across shops and weighted with
Fellegi–Sunter weights learned on training data. Title and key tokens are
hashed together in the one index (title + keys). After LSH, pairs from the same shop or of two different
known brands are removed, and the remaining pairs are classified with a
logistic regression followed by a transitivity step. The one index (title +
keys) reaches the F1 of classifying all pairs (.600) with 0.5% of the
comparisons; MSMP reaches it only at its 250 × 2 setting, with 62% of them.

## Data

`data/TVs-all-merged.json`: 1,624 television listings from four Web shops
(Amazon, Best Buy, Newegg, TheNerds), with title, shop, key-value pairs and
the modelID. Two listings are duplicates if their modelIDs are equal: 399
duplicate pairs.

## Installation and reproduction

Python 3.11 (tested with 3.11.9).

```bash
pip install -r requirements.txt
for t in tests/test_*.py; do python $t; done   # tests, about 20 s
python run_experiment.py --jobs 4              # 10 bootstraps, about 5 min (about 100 s per bootstrap)
python make_tables.py                          # tables, a few seconds
python make_plots.py                           # figure, a few seconds
```

| Paper | Output |
|---|---|
| Table 1 | `results/table1.csv` |
| Table 2 (Appendix B) | `results/table2.csv` |
| Table 3 (Appendix B) | `results/table3.csv` |
| Figure 1 | `paper/figures/results_curves.pdf` (copy: `results/results_curves.png`) |
| Values quoted in Sect. 4 (F1 and PC at 0.1%, 0.5%, 1%, 2% of comparisons) | `results/f1_at_fractions.csv` |

`results/raw_results.csv` holds every bootstrap, method and setting,
`results/summary.csv` the means and standard deviations over the bootstraps,
and `results/learned_per_bootstrap.json` the training choices per bootstrap
(c of our methods and of the benchmarks, s, s_k, the scales of the one index
(title + keys), selected keys, key groups, probability thresholds and MSM
parameters).

## Method

**3.1 Title tokens.** Titles are lowercased, inch and hertz notations are
unified and punctuation is removed except decimal points; tokens of at least
three characters that are part of the shop's name are dropped. Tokens with
document frequency below 2 or above c·N are removed; c ∈ {0.1, 0.2, 0.3, 0.5}
maximises training F1\*.

**3.2 Key-value tokens.** Values are normalised (first rounded number for
numeric keys, lowercase alphanumerics otherwise). Keys of different shops are
merged when their values agree for training duplicates far more often than for
non-duplicates (|D| ≥ 10, m ≥ 0.8, log(m/u) ≥ 1, add-one smoothing, connected
components). Per canonical key the Fellegi–Sunter weights w⁺ = log(m/u) and
w⁻ = log((1−m)/(1−u)) are estimated; the K = 10 keys with the highest w⁺ give
`key=value` tokens.

**3.3 Graded weights.** A token with weight w becomes max(1, round(scale·w))
distinct copies, so MinHash estimates the weighted Jaccard similarity. Title
tokens are weighted by idf, key tokens by w⁺; the scales s and s_k ∈ {0.5, 1, 2}
maximise training F1\*.

**3.4 MinHash and LSH.** n = 500 hash functions (a·x + b) mod (2³¹ − 1) on
CRC32 token ids; for r rows per band, b = ⌊500/r⌋ bands, 14 settings from
500 × 1 to 25 × 20. Three methods: the title index, the key index, and the
one index (title + keys), which hashes the title and key tokens together
(prefixes `t:` and `k:`).

**3.5 Filters, classifier and transitivity.** Candidate pairs from one shop
or of two different known brands are removed (our three methods only). A
logistic regression on five features (title token Jaccard, title character
3-gram Jaccard, idf cosine on all title tokens, S⁺ and S⁻) classifies the
remaining pairs. Predicted pairs are merged into clusters in order of
decreasing probability, rejecting merges that put two products of one shop or
of different known brands together. The probability threshold maximises
training F1 with this step applied.

**3.6 Evaluation and bootstrap.** 10 bootstraps; the training set is the
distinct drawn products (about 63%), the test set the products never drawn.
Everything is learned on the training set; PQ, PC, F1\*, precision, recall and
F1 are measured on the test set and averaged.

**Benchmarks.** *MSMP with our classifier*: the title model words of MSMP
(raw titles) in the same LSH, without the filter, classified by our logistic
regression (6 features, with `same_shop`). *MSMP with MSM*: the same
candidates decided by our reproduction of MSM (multi-component similarity and
adapted single-linkage clustering). *All pairs*: our classifier (6 features)
and MSM on all pairs, without LSH.

Assumptions that the paper does not spell out are listed in
[ASSUMPTIONS.md](ASSUMPTIONS.md).

## Repository structure

```
README.md                 this file
.gitignore                files kept out of git
ASSUMPTIONS.md            assumptions behind the final approach
requirements.txt          Python packages
run_experiment.py         runs the 10 bootstraps; writes raw_results, summary, learned_per_bootstrap
make_tables.py            writes table1-3.csv and f1_at_fractions.csv
make_plots.py             writes Figure 1
data/                     TVs-all-merged.json
src/dupdet/config.py      fixed settings
src/dupdet/data.py        loading the data, title cleaning (3.1)
src/dupdet/representation.py  df filter, idf, graded copies, one-index tokens, brands, MSMP model words
src/dupdet/keys.py        value normalisation, key alignment, Fellegi-Sunter weights, key selection (3.2)
src/dupdet/lsh.py         MinHash and banding (3.4)
src/dupdet/features.py    pair features (3.5)
src/dupdet/classifier.py  logistic regression and threshold (3.5)
src/dupdet/transitivity.py  transitivity step (3.5)
src/dupdet/msm.py         MSM benchmark
src/dupdet/evaluate.py    PQ, PC, F1*, precision, recall, F1 (3.6)
src/dupdet/pipeline.py    one bootstrap from split to test results
tests/                    tests of the bootstrap, filters, graded weights, LSH, MSM and transitivity
results/                  output of the run (see the table above)
paper/figures/            results_curves.pdf (Figure 1)
```

## Implementation note

MinHash and LSH (hash functions, signatures and banding) are implemented from
scratch in `src/dupdet/lsh.py`; no LSH library is used. scikit-learn is used
only for the logistic regression and rapidfuzz only for Levenshtein distances
in MSM.

## AI statement
The code in this repository was written with the help of AI (Claude,
Anthropic). All decisions on the research design, methods, parameters,
evaluation and interpretation of the results were made by me, Borg
Knijnenburg.
