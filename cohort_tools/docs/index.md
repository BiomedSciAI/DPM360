# Overview

`cohort_tools` contains code for the extraction of features related to cohorts defined via ATLAS or custom queries and used for developing deep learning DPM algorithms in `lightsaber` using Python.

`cohort_tools` comprises of `cohort_connector` and `feature_extractor`.

`lightsaber` integrates naturally with ATLAS using a client called `cohort_connector`, enabling automated extraction of features from the CDM  model, thus complementing the ease and flexibility of defining standardized cohorts using ATLAS graphical user interface with the ability to quickly develop deep learning algorithms for DPM in `lightsaber` using Python.

Once `cohort_connector` has been configured with database credentials, `feature_extractor` can be configured with the cohort details, covariate settings to extract the right set of features in formats currently supported in the OHDSI stack  and PatientLevelPrediction R packages via the Rpy2 interface.

Additionally, the `feature_extractor` uses custom queries and algorithms to extract and transform complex time series features into formats required for DPM in `lightsaber`. For each feature extraction process, a YAML configuration file is automatically generated. This file specifies outcomes, covariate types, and file locations of the extracted feature files. 

Thus, subsequently, `lightsaber` allows a user to concentrate just on the logic of their model as this component takes care of the rest.
