# API Documentation

## Data Ingestion


### ::: lightsaber.data_utils.pt_dataset.BaseDataset
    selection:
      members: __init__
    rendering:
      show_source: false

### ::: lightsaber.data_utils.pt_dataset.collate_fn
    rendering:
      show_source: true

### ::: lightsaber.data_utils.sk_dataloader.SKDataLoader
    selection:
      members: __init__

### Filters and Transforms

#### ::: lightsaber.data_utils.pt_dataset.identity_2d
    rendering:
      show_source: true

## Model Training

### ::: lightsaber.trainers.pt_trainer.PyModel
    selection:
      members: __init__

### ::: lightsaber.trainers.pt_trainer.run_training_with_mlflow
    rendering:
      show_source: false
      show_root_heading: true

### ::: lightsaber.trainers.sk_trainer.SKModel
    selection:
      members: __init__


### ::: lightsaber.trainers.sk_trainer.run_training_with_mlflow
    rendering:
      show_source: false
      show_root_heading: true

## Model Registration and Load

### PyTorch

#### ::: lightsaber.trainers.pt_trainer.register_model_with_mlflow
    rendering:
      show_source: false
      show_root_heading: true
  
#### ::: lightsaber.trainers.pt_trainer.load_model_from_mlflow
    rendering:
      show_source: false
      show_root_heading: true

### Scikit-learn

#### ::: lightsaber.trainers.sk_trainer.register_model_with_mlflow
    rendering:
      show_source: false
      show_root_heading: true
  
#### ::: lightsaber.trainers.sk_trainer.load_model_from_mlflow
    rendering:
      show_source: false
      show_root_heading: true
