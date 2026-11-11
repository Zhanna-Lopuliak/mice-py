# Regenerate every reference fixture. Run from the repository root:
#   Rscript tests/reference/r/generate_all.R
#
# Each generator is a file named l<layer>_<topic>.R in this folder and runs in
# its own environment, so the scripts cannot depend on each other.

scripts <- sort(list.files(
  file.path("tests", "reference", "r"),
  pattern = "^l[0-9]_.*\\.R$", full.names = TRUE
))
for (script in scripts) {
  message("== ", script)
  source(script, local = new.env())
}
