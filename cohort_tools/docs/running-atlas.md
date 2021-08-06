# Running Atlas-Lightsaber 

The cohort_tools folder contains code related to the extraction of features related to cohorts defined via ATLAS or custom queries and used for developing deep learning DPM algorithms in Lightsaber using Python.

The inhospital_mortality_mimic_iii folder under the demos folder contains Jupyter notebooks illustrating code related to running specific models for prediction tasks. 

These are the steps required to run the feature extraction from Atlas and the training/registration of the model.

<b>Step 1: Running feature extraction to generate files needed for training model with Lightsaber</b>

<b>Pre-requisites:</b>
Ensure the environment is setup using the requirements.txt file in the cohort_tools folder
Some of the pre-requisites may need to be modified to suit your environment. 
Specifically you would need to install R, and the rpy2 version may need to be modified depending on your OS. 
The requirements file provided are more as a guideline and not definitive.

<b>Execution:</b>

1. Ensure that you are using the environment created using the requirements file. 

2. Run the IHM_MIMIC_III_feature_extraction.ipynb notebook from demos folder

	1. Ensure you know the unique identifiers of the target and outcome cohorts defined in ATLAS. Optionally, generate new cohorts using custom queries as illustrated in the IHM_MIMIC_III_custom_cohort_definition.ipynb Jupyter notebook in the demos file

	2. Create an instance of the CohortConnector class by specifying the connection details via a json file or passing the required arguments. This object is used to connect to specific target and outcome cohorts in an OMOP CDM database

	3. Create an instance of the FeatureExtractor class by passing a previously created CohortConnector object as an argument along with the feature extraction settings specified in a json file or passed as arguments. This will be used to extract features from the cohorts.

	4. Extract features for training using the 'extract_features' function and specifying setup='train' as an argument to the function.  
	5. Extract features for prediction using the 'extract_features' function and specifying setup='prediction' as an argument to the function.


<b>Outputs:</b>

1. Data folder containing CSV files for model training and validation 
These map to the features and output for train and test files

2. Generated experiment configuration (YAML) file. 


<b>Step 2: Training and registering model using features from atlas</b>

Pre-requisites:
Run the conda.yaml file to setup the environment
Export the following variables with the appropriate values, these need to be set prior to running the notebook in
export AWS_ACCESS_KEY_ID=''
export AWS_SECRET_ACCESS_KEY=''
export MLFLOW_S3_ENDPOINT_URL=''
export MLFLOW_URI=''
 
Execution Via Jupyter notebook
Run the Exp_LSTM.ipynb from demos folder after setting the following variables

1. Ensure the configurations for MLFlow are setup correctly

2. Ensure Minio password/username/url/other credentials are setup correctly

3. Make sure the lightsaber mlflow path is setup correctly so that the registration/logging of model works

Outputs of this step are:
You should be able to see the model created in the MLFlow UI
If you logged any artifacts, they would be available in the MLFlow UI as well