# Layer 2 reference for the donor methods with general masks: pmm,
# midastouch, cart, rf and sample on input F8 (see _l2_common.R), called as
# sampler.univ calls them, mice.impute.<method>(y, ry, x, wy), where the rows
# to impute include 8 fit rows and 8 rows lie in neither mask. Each method
# runs with the arguments and the number of calls of its fixture on F1 to F4
# (l2_<method>.R), after one set.seed, and stores the same statistics; the
# cart fixture adds `reached_2000`. A row in both masks can be its own donor;
# there is no simple theory for that, so R is the reference.
#
# Every imputation must be the value of a fit row (run_calls() stops
# otherwise), which also shows that R never draws a decoy y of a row in
# neither mask.
#
# Writes tests/reference/fixtures/l2_masks_pmm.json, l2_masks_midastouch.json,
# l2_masks_cart.json, l2_masks_rf.json and l2_masks_sample.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

# donors = 5, matchtype = 1, quantify = TRUE
l2_method_fixture("pmm", input_names = "F8", seeds = 1008, n_calls = 20000)

l2_method_fixture("midastouch", input_names = "F8", seeds = 2008, n_calls = 20000)

# minbucket = 5, cp = 1e-4
l2_method_fixture(
  "cart",
  input_names = "F8",
  seeds = 3008,
  n_calls = 20000,
  extra = reached_first_2000
)

# ranger, ntree = 10, on one thread so that the file is reproducible
l2_method_fixture("rf", input_names = "F8", seeds = 4008, n_calls = 5000, num.threads = 1)

l2_method_fixture("sample", input_names = "F8", seeds = 5008, n_calls = 20000)
