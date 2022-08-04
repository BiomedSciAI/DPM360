# Release History

Release notes for major `lightsaber` releases are linked in this page

## v0.3.0

### Highlights

- upgraded `pytorch_lightning` to version `1.6.4`
- updated test script, api largely backported
- Updated example for LSTM/HistGBT

### Changes in API:

- `pl_trainer.run_training_with_mlflow`: 

  - Call Signature updated

    - dataloaders can be now passed directly here instead of in wrapped model
    - instead of instantiated trainer, arguments supported by pytorch lightning Trainer is passed as Namespace/ArgumentParser

  - Return type updated. validation/test input and outputs (e.g. `y_val`, `y_pred`) are now numpy not torch tensors


## v0.2.6

- added test data generator 
- added new test cases for classification in `./tests/Test_Classification_*.ipynb`
- bug fixes: make `feat_columns` specification more fault tolerant w.r.t. `time_order_col` 

## v0.2.5

- bug fixes from `v0.1`
- added `sk_trainer` example
- added new test case in `./tests/test_dataset.py`
- better mlflow support - model trainings now return run id 
