# In Hospital Mortality Model

Example to model IHM from MIMIC-III benchmark data.

**Pre-requisities**

* Get access to MIMIC-III data
* Run the data generation component of [mimic-iii benchmark code](https://github.com/YerevaNN/mimic3-benchmarks)

**Running the examples**

* Run the [Pre-processing notebook](./PreprocessMimicBenchmark.ipynb) to convert the train/validation/test split into csv forms that can be ingested by `lightsaber` functions
* Running the models:
  * `HistGBT` model: [notebook](./Exp_HistGBT.ipynb)
  * `LSTM` model: [notebook](./Exp_LSTM.ipynb)
