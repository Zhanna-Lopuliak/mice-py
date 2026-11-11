# L1 fixture for quickpred().
#
#   l1_quickpred.json   quickpred(data, ...) on nhanes and boys with several
#                       argument sets
#
# Python target: the planned imputation.quickpred.
# quickpred() works on data.matrix(data): factors enter as their integer
# codes in level order, so the fixture keeps R's level order (boys$reg is
# north, east, west, south, city, not alphabetical).

source(file.path("tests", "reference", "r", "_common.R"))

argument_sets <- list(
  default = list(),
  mincor_0.3 = list(mincor = 0.3),
  minpuc_0.25 = list(minpuc = 0.25),
  include_exclude = NULL, # filled per data set below
  spearman = list(method = "spearman")
)
include_exclude <- list(
  nhanes = list(include = "hyp", exclude = "age"),
  boys = list(include = "reg", exclude = "tv")
)

cases <- list()
for (data_name in c("nhanes", "boys")) {
  data <- get(data_name)
  for (arg_name in names(argument_sets)) {
    args <- if (arg_name == "include_exclude") {
      include_exclude[[data_name]]
    } else {
      argument_sets[[arg_name]]
    }
    pm <- do.call(quickpred, c(list(data), args))
    full <- list(
      mincor = 0.1, minpuc = 0, include = character(0),
      exclude = character(0), method = "pearson"
    )
    full[names(args)] <- args
    cases[[length(cases) + 1L]] <- list(
      name = paste(data_name, arg_name, sep = "_"),
      data_name = data_name,
      mincor = full$mincor,
      minpuc = full$minpuc,
      include = I(full$include),
      exclude = I(full$exclude),
      method = full$method,
      rows = I(rownames(pm)),
      cols = I(colnames(pm)),
      predictor_matrix = unname(pm)
    )
  }
}

write_fixture("l1_quickpred", list(
  description = paste(
    "quickpred(data, mincor, minpuc, include, exclude, method). Rows are",
    "targets, columns predictors. Factors are encoded with R's level order."
  ),
  data = list(nhanes = encode_frame(nhanes), boys = encode_frame(boys)),
  cases = cases
))
