# Overview


At a high level, there are four components for `lightsaber`:

* `Datasets`: to support standardized methods of ingesting data modules
* `trainers`: to support standardized model training using best practices
* `metrics`: to expose pre-builts DPM problem specific model evaluation
* In-built model tracking and support for post-hoc model evaluation using MLFLow

We next go through each of these below


## Data Ingestion Modules


The primary data ingestion is provided by `lightsaber.data_utils.pt_dataset.BaseDataset` class

<!-- ::: lightsaber.data_utils.pt_dataset.BaseDataset -->

It accepts the following parameters

```
Parameters
----------
tgt_file:
    target file path
feat_file:
    feature file path
idx_col:
    columns to specify the unique examples from the feature and target set
tgt_col:
    columns to specify the target column from the target set.
feat_columns:
    feature columns to select from. either list of columns (partials columns using `*` allowed) or a single regex
    Default: `None` -> implies all columns
time_order_col:
    column(s) that signify the time ordering for a single example.
    Default: `None` -> implies no columns
category_map:
    dictionary of column maps
transform: single callable or list/tuple of callables
    how to transform data. if list of callables provided eg `[f, g]`, `g(f(x))` used
    Default: drop `lightsaber.constants::DEFAULT_DROP_COLS` and fillna
filter: single callable or list/tuple of callables
    how to filter data. if list of callables provided eg `[f, g]`, `g(f(x))` used
    Default: no operation
device: str
    valid pytorch device. `cpu` or `gpu`
```

Specifically, `feat_columns` can take either a list of feature columns to be used or reg-ex specifying the sub-list of columns.
It can also support a hybrid approach such as:

```python
feat_columns = ['featA', 'b_feat*']
```

### Filters and Transforms

In addition, `BaseDataset` supports a set of compossible functions called `filters` and `transforms` that can be used
to transform the input data.

**Filters** accepts as arguments `(data, target)` and other keyword functions and always returns `(data, target)`.
An example filter to fill `NA` from the feature dataset is as follows:

```python
@ptd.functoolz.curry
def filter_fillna(data, target, fill_value=0., time_order_col=None):
    data = data.copy()

    idx_cols = data.index.names
    if time_order_col is not None:
        try:
            sort_cols = idx_cols + time_order_col
        except:
            sort_cols = idx_cols + [time_order_col]
    else:
        sort_cols = idx_cols

    data.update(data.reset_index()
               .sort_values(sort_cols)
               .groupby(idx_cols[0])
               .ffill())

    data.fillna(fill_value, inplace=True)

    return data, target

```


Lightsaber comes pre-packaged with a set of filters:

* `filter_fillna`: fill `NA`
* `filter_preprocessor`: to chain any `sklearn` pre-processor in the correct manner
* `filter_flatten_filled_drop_cols`: to flatten temporal data, fill `NA`, and drop extra columns
* `filt_get_last_index`: to get the last time point from each example for a temporal dataset (e.g. useful for training `Med2Vec` models)


In addition, filters can be defined at run-time by a user


**Transform** are functions that are applied at run-time while returning the data for a single
example. it accepts as arguments `(data)` and other keywords. It always returns `(data)`.

Lightsaber comes pre-packaged with a set of transforms including flattening and dropping `NA` at runtime.

_transforms_ are generally discouraged as these are applied at run time and can slow down data fetching.
In general, if it is possible to load the entire data in memory use `filters` - else use `transforms`.


### Helpers for scikit-learn

While `BaseDataset` is a general purpose data ingestion module, `lightsaber`
provides a higher level api to access data in a format more accessible to people familiar with `scikit-learn`.

It is provided by `lightsaber.data_utils.sk_dataloader.SKDataLoader` class:

It accepts the following parameters:

```
Parameters
----------
tgt_file:
    target file path
feat_file:
    feature file path
idx_col:
    columns to specify the unique examples from the feature and target set
tgt_col:
    columns to specify the target column from the target set.
feat_columns:
    feature columns to select from. either list of columns (partials columns using `*` allowed) or a single regex
    Default: `None` -> implies all columns
time_order_col:
    column(s) that signify the time ordering for a single example.
    Default: `None` -> implies no columns
category_map:
    dictionary of column maps
fill_value:
    pandas compatible function or value to fill missing data
flatten:
    Functions to aggregate and flatten temporal data
cols_to_drop:
    list of columns to drop
preprocessor:
    any scikit-learn pre-processor that needs to be used agains the data
```

Example usage:

```python
from lightsaber import constants as C
from lightsaber.data_utils import sk_dataloader as skd

flatten = C.DEFAULT_FLATTEN

train_dataloader = skd.SKDataLoader(tgt_file=expt_conf['train']['tgt_file'],
                                    feat_file=expt_conf['train']['feat_file'],
                                    idx_col=idx_col,
                                    tgt_col=tgt_col,
                                    feat_columns=feat_cols,
                                    category_map=category_map,
                                    fill_value=fill_value,
                                    flatten=flatten,
                                    preprocessor=preprocessor)
fitted_preprocessor = train_dataloader.get_preprocessor(refit=True)

val_dataloader = skd.SKDataLoader(tgt_file=expt_conf['val']['tgt_file'],
                                  feat_file=expt_conf['val']['feat_file'],
                                  idx_col=idx_col,
                                  tgt_col=tgt_col,
                                  feat_columns=feat_cols,
                                  category_map=category_map,
                                  fill_value=fill_value,
                                  flatten=flatten,
                                  preprocessor=fitted_preprocessor)
```



## Model Training

Model Training is supported for both `pytorch` and `scikit-learn` models.


### Scikit-learn models


For scikit-learn, a simplified model training and hyper-parameter tuning framework is exposed via `ligthsaber`.

It is provided by `lightsaber.trainers.sk_trainer`.

The three important functions are as follows:

`SKModel`: A wrapper that connects eng backends to standard scikit-learn compatible models.
It accepts the following parameters:

```
Parameters
----------
base_model:
  any scikit-learn compliant model (e.g. models subclassing `BaseEstimator`)

model_params:
  model hyper-params to init the model
name: OPTIONAL
  name to identify the model
```

Some of the important functions provided by `SKModel`:

* `fit`
* `calibrate`: to provide post fitting calibration from calibration dataset
* `tune`: run hyper-parameter training using `GridSearchCV`


**Other improtant functions**

`lightsaber.trainers.sk_trainer.model_init`: helper function to abstract standard model training pre-amble for training

`lightsaber.trainers.sk_trainer.run_training_with_mlflow`: run training of sklearn models with model tracking

It accepts the following parameters:

```
Parameters
----------
mlflow_conf:
    configurations to connect to mlflow. e.g. default mlflow uri
sk_model:
    an instance of `SKModel`
train_dataloader:
    an instance of `SKDataLoader` for training dataset
val_dataloader: OPTIONAL
    an instance of `SKDataLoader` for validation dataset
test_dataloader: OPTIONAL
    an instance of `SKDataLoader` for test dataset
kwargs:
    other optional keywords such as `tune`
```

### Pytorch Models

For `pytorch` models a trainer using the SOTA methods is provided `lightsaber.trainers.pt_trainer`

The most important modules is `PyModel`. It takes the following parameters


```
hparams:
  hyper-paramters. instance of NameSpace
model:
  any standard pytorch-models
train_dataset:
val_dataset:
test_dataset: OPTIONAL
cal_dataset: OPTIONAL
collate_fn: OPTIONAL
    collate function to process the temporal data e.g. `ptd.collate_fn`
optimizer: OPTIONAL
loss_func: OPTIONAL
out_transform: OPTIONAL
num_workers:
```
