# Assumptions

Assumptions behind the final approach that the paper does not spell out.
Section numbers refer to the paper. Numbers come from the included run
(seed 2026, 10 bootstraps).

## 1. Data

**1.1 Duplicates are defined by the modelID.** Two listings are duplicates if
and only if their modelIDs are identical: 399 pairs among 1,317,876. The
modelID is used only for training labels and evaluation. The `url` field is
not loaded, because Amazon URLs contain the model ID. The grouping of the
JSON file by modelID is discarded; products are sorted by shop, title and URL.

**1.2 Identical listings with different modelIDs.** There are 4 groups (10
listings, all Newegg) with the same shop, title and features but a different
modelID each, for example "Newegg.com - Samsung 55" 1080p LED 3D TV" for
UN55F6400, UN55F8000BF and UN55F7100. By the gold standard they are different
products that no method can tell apart; as same-shop pairs they are removed by
every same-shop rule. They are kept as they are.

**1.3 The 12 duplicate pairs within one shop.** 12 of the 399 duplicate pairs
(11 modelIDs; 8 at Newegg, 4 at Best Buy) lie within one shop, for example a
second listing with a "Newegg.com -" prefix or a refurbished listing. They
stay in the gold standard, but no method can find them: our filter, the
transitivity step and MSM all reject same-shop pairs. They count as false
negatives in PC and F1. `tests/test_bootstrap.py` checks 1.2 and 1.3.

## 2. Bootstrap (Sect. 3.6)

**2.1 No repeated products.** Each bootstrap draws 1,624 indices with
replacement. The training set is the set of distinct drawn products (about
63%), so a product drawn twice appears once; otherwise it would form a false
duplicate pair with itself. The test set is the products never drawn
(out-of-bag). Duplicate pairs with one product in each set belong to neither:
a training set has about 157 duplicate pairs, a test set about 53.

**2.2 What may use test data.** Everything that needs no labels is computed
on the set it is applied to: document frequencies, idf weights and the brand
vocabulary of the test set come from the test products. Everything that needs
labels (c, s, s_k, key alignment, key weights, key selection, the classifier
and its threshold, the MSM parameters) is learned on the training set only.

**2.3 Randomness.** Each bootstrap has its own random generator, seeded with
(2026, bootstrap number); it draws the split, the MinHash functions and the
non-duplicate samples of the key model. The same seed gives identical results.

## 3. Representations and LSH (Sect. 3.1–3.4)

**3.1 Two cut-offs c.** c is chosen once per bootstrap: for each c the best
training F1\* over the 14 settings of the df-filtered title tokens (before the
graded copies), on the training candidates after the same-shop and brand
filter. This c is used by the title index, the one index and the classifier
features of our methods. The classifier features of the unfiltered benchmarks
(MSMP with our classifier, all pairs) use a second c, chosen the same way but
with only the brand rule, so that the benchmarks do not depend on our filter.

**3.2 Ties.** Ties in training F1\* go to the first value in the grid (c, s,
s_k; for the one index the pair (s, s_k) in the order s, then s_k); ties in
training F1 go to the lowest probability threshold.

## 4. Key-value tokens (Sect. 3.2, Appendix A)

**4.1 Hand-set alignment thresholds.** |D| >= 10, m >= 0.8 and log(m/u) >= 1
were set by hand and not tuned. For the alignment, u is estimated on the
cross-shop pairs among 20,000 sampled training non-duplicate pairs; for the
weights m and u of Sect. 3.2, u is estimated on 50,000 sampled training
non-duplicate pairs. Both use add-one smoothing (agree + 1) / (n + 2).
A key needs at least 10 duplicate pairs with both values present to get a
weight. The thresholds do not stop every wrong merge: in 3 of the 10
bootstraps a free-text key ("meta keywords", "meta description") is merged
into screen size, because its first number is the screen size. It does little
harm, because within a merged group the value of the most frequent key name
is used.

**4.2 Value normalisation.** A key is numeric if at least 80% of its training
values contain a digit; its values are reduced to the first number, rounded
to an integer (`46" Class (45.9" Diag.)` becomes `46`; `1920 x 1080` becomes
`1920`). Other values are reduced to lowercase letters and digits. Units are
not converted. Different values can therefore agree; a key whose values agree
too often for non-duplicates gets a low weight and is not selected.

## 5. Filters, classifier and transitivity (Sect. 3.5)

**5.1 Comparisons are counted after the filters.** For our three methods the
same-shop and brand filter acts on the LSH candidates before they are counted,
so the number of candidates, the fraction of comparisons, PQ and PC are those
of the filtered candidates. Checking the shop or the brand of a pair is not
counted as a comparison. The denominator of the fraction stays n(n−1)/2. The
benchmarks are not filtered; the classifier on all pairs compares every pair.

**5.2 Brand list.** A product's brand is the first title token found in the
brand vocabulary, else the first word of its own brand value, else unknown.
The vocabulary is the first word of every value of a key whose name contains
"brand", plus a hard-coded list of eleven TV brands that occur only in titles
of listings without a brand key: avue, azend, contex, curtisyoung, dynex, gpx,
hiteker, insignia, tcl, venturer, viore. With the list, 2 products have an
unknown brand instead of 82. No duplicate pair has two different known
brands, so the brand rule removes no true duplicate (`tests/test_blocking.py`).

**5.3 Classifier details.** Logistic regression of scikit-learn (L2 penalty,
default C) on standardised features, fitted separately per method, setting
and bootstrap. The probability threshold is taken from 0.05, 0.10, ..., 0.95.
Without any duplicate among the training candidates nothing is predicted;
with only duplicates every candidate is predicted. For our methods
`same_shop` is always 0 and is left out (5 features). The transitivity step
uses the shop rule for every method and the brand rule for our methods.
Pairs added by the transitivity step were never compared and are not counted
as comparisons.

## 6. MSM benchmark

**6.1 Reproduction choices.** The papers leave several choices open; ours:

- Preprocessing: lowercase; `( ) [ ] , ; : ! ? + & * /` become a space
  (deleting them instead would create false model words such as `(2012` and
  `(46`, and turn `49-1/2"` into `4912"`); `.`, `"` and `-` are kept.
- Exclusions: infinite distance for pairs from one shop and pairs of two
  different known brands, using our brand rule (5.2) instead of a hand-made
  brand list; for MSMP with MSM also for pairs that are not LSH candidates.
  For MSM on all pairs, `raw_results.csv` and `summary.csv` report the pairs
  without infinite distance (about 15,230 per test set) as its comparisons;
  Table 1, like the classifier row, reports all 182,324 pairs.
- q-grams: q = 3, strings padded with two dummy characters at both ends.
- Key matching: greedy and one-to-one; the key-value pairs of the product with
  the lower index are visited in data order, each matched with the first
  unmatched pair of the other product with key similarity above gamma.
- Model words of values: as in MSMP+, the numeric part of tokens like `46inch`
  or `46`; mwPerc is their Jaccard similarity. This option was selected on the
  mean training F1 over the bootstraps.
- Title similarity: the two weights of 0.5 between cosine, average
  Levenshtein similarity and model-word similarity are ours; the cosine is
  rounded to 12 decimals before it is compared with alpha.
- Parameters: beta = 0 and q = 3 fixed; alpha, gamma, mu and epsilon from
  0.0, 0.1, ..., 1.0, chosen per bootstrap (and per LSH setting for MSMP with
  MSM) by training F1, where training duplicates outside the pairs count as
  false negatives. Ties go to the first value in grid order.

**6.2 Why MSM stays below its published F1.** Likely reasons, none tested
separately: the open choices above differ from the original implementation;
our brand rule replaces the original brand list; the parameters are tuned on
a training set of about 63% of the products with about 157 duplicate pairs and
evaluated on out-of-bag test sets with about 53, where the within-shop
duplicates (1.3) count as false negatives. The comparison with MSM is
therefore less reliable than the comparison with MSMP's LSH step.
