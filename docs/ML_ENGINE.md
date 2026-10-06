# ML Engine

## Classical engine

The preserved scikit-learn workbench supports classification, regression and clustering with explicit dataset, features, target, preprocessing, algorithm, hyperparameters and seed. Results include train/test metrics, confusion data where relevant, diagnostics and reproducibility metadata. Experiment history and comparisons continue to use the original run schema.

## Dataset sources

### Bundled

Synthetic and scikit-learn datasets are registered in `datasets.py`. They are deterministic and available offline.

### Personal uploads

`user_datasets.py` accepts CSV, TSV, JSON records and XLSX. Upload handling enforces:

- 10 MiB compressed/source limit;
- 50 MiB expanded XLSX limit;
- 50,000 rows, 200 columns and 20,000 characters per cell;
- extension/content checks and one worksheet;
- no macro formats, formulas, pickle or object deserialization;
- UUID storage and player ownership;
- normalized CSV persistence;
- actionable warnings for duplicate columns, missing data and spreadsheet formulas.

Formula cells are never evaluated. User-provided paths are never used for storage.

## Split and leakage discipline

Uploaded-data training resolves the target and features server-side. Preprocessing is fitted in the training pipeline rather than over the complete dataset. Classification splits are stratified when class counts permit. Validation catches nonexistent columns, unsupported tasks/models, unsuitable target cardinality and too-small data.

The engine reports seeds and diagnostics; it does not claim that one held-out score proves generalization. Cross-validation remains available in the original workbench where supported.

## Reproducibility

A saved run contains:

- source dataset identifier;
- target/features;
- task and model;
- preprocessing and hyperparameters;
- seed and context;
- measured result and summary;
- timestamp and learner notes.

The deterministic top-five score utility is `neural_forge.ranking.top_five_scores`. It groups labeled finite scores, keeps each label’s best score, applies a deterministic label tie-break and returns at most five records.

## Boundaries

The classical engine is not AutoML. It does not silently search a broad hyperparameter space, choose a target or claim causal conclusions. Uploaded arbitrary serialized estimators are not accepted.
