
# DPM360
Repository for Disease Progression Modeling workbench 360

Overview and YouTube demonstration are available [here](https://ibm.github.io/DPM360/). License, Contribution, Publications are also available there.

<figure><center><img src=./docs/resources/png/dpm360v2.png "DPM360" width="300"/></center><figcaption>DPM360 Component View</figcaption></figure>

# Installation Guides

DPM360 components are interoperable but can also work as independent tools. DPM360 is typically installed over a cluster that sets up a number of interconnected micro-services. We can broadly divide the components into two groups viz (i) components that are concerned with micro service setups and (ii) standalone python packages providing core functional capabilities, having separate installation procedures. Please see the guides below to install each component.

## DPM360 micro-service utilities

One of the key micro-service utilties is <b>installer</b> that sets up an OHDSI stack (Atlas, WebAPI, a Postgres Database, and Achilles) into a cloud cluster such as Kubernetes or OpenShift. See [installation guide](https://github.com/IBM/DPM360/blob/main/installer/docs/installer.md) for details. Its [<b>Express Installation Script</b> section](https://github.com/IBM/DPM360/blob/main/installer/docs/installer.md#express-installation-script) provides minimum setup operations. Using this component:
- run a OMOP CDM database using Postgres on your cloud cluster
- run Atlas, WebAPI and other OHDSI service with the DB
- run Model Registry using MLFlow where your learned models are registered

The <b>service builder</b> component packages and deploys the learned models to the target cloud cluster. See [installation guide](https://github.com/IBM/DPM360/blob/main/service_builder/docs/README.md) for details. Using this component:
- make a microservice by deploying the model registered in Model Registry using KFServing
- test and interact with the deployed model microservice via a Swagger based interface

## DPM360 standalone python packages - enabling deep learning model training in python on OHDSI-OMOP data

The [<b>lightsaber</b>](https://github.com/IBM/DPM360/blob/main/lightsaber/docs/index.md) component is an extensible Python training framework which provides blueprints for the development of disease progression models. See [installation guide](https://github.com/IBM/DPM360/blob/main/lightsaber/docs/install.md). Also see [user guide](https://github.com/IBM/DPM360/blob/main/lightsaber/docs/user_guide.md) for data loading and training details. Using this component:
- develop machine learning models using extensible data loaders and training pipelines
- use extensible data loaders designed for time-series dataset extracted from OHDSI and other EMRs
- use scikit-learn and PyTorch Lightning based training pipelines with pre-defined networks and loss functions for processing time-series dataset.
- save and register your learned models with experimental artifacts in Model Registry

The [<b>cohort tools</b>](https://github.com/IBM/DPM360/blob/main/cohort_tools/docs/index.md) component provides python scripts to extract features from cohorts defined via ATLAS or custom queries. It enables [integration with lightsaber](https://github.com/IBM/DPM360/blob/main/cohort_tools/docs/user_guide.md) to use features extracted from OHDSI databases. Using this component:
- with cohort defined by Atlas and others, define and extract the features from the OMOP CDM database
- extract such features in CVS files and use them as inputs for lightsaber 
