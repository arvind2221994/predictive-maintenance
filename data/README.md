# Training dataset setup

Place your actual, permitted dataset in this directory with the name `engine_data.csv`.

The public repository does not currently contain a dataset, and this deliverable does not invent or include one. Without the CSV, the pipeline intentionally fails at data preparation rather than reporting a successful training run.

## Required CSV columns

| Exact column name | Required values |
| --- | --- |
| Engine rpm | Finite, non-negative numeric RPM. |
| Lub oil pressure | Finite, non-negative numeric pressure. |
| Fuel pressure | Finite, non-negative numeric pressure. |
| Coolant pressure | Finite, non-negative numeric pressure. |
| lub oil temp | Finite numeric temperature; negative and unusually high values are not clipped. |
| Coolant temp | Finite numeric temperature; negative and unusually high values are not clipped. |
| Engine Condition | Both numeric class labels `0` and `1`. Confirm their meanings from the data source. |

Use the same units as the notebook and existing app: RPM, bar, and degrees Celsius. The column spelling and capitalization must match. Additional ID/index/timestamp columns are not used as model features.

Missing, infinite, non-numeric, invalid-label, and negative RPM/pressure values cause explicit errors. Duplicate feature-and-target rows are removed. After deduplication, at least five records in each class are needed for the stratified 80/20 split and training-only 3-fold cross-validation.

**The existing GitHub repository is public. Only commit this dataset if you are authorized to make it public.** Do not include personal/customer information or credentials. For private data, use an approved private repository/data-access workflow instead of publishing it here.

See the [main setup instructions](../README.md) for local training, tests, GitHub Actions, and publishing. Intermediate prepared/train/test CSVs are written to the ignored `artifacts` directory, not committed or uploaded as deployment artifacts.
