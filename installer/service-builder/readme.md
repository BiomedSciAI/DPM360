**to be updated

Create cron job by execution cronjob.yaml file from the container platfor UI

The following paramters need to be populated in the cronjob.yaml file, prior to execution. 

K8s_CLUSTER : This is cluster id. 
K8S_API_KEY: API key with access to k8s / oc cluster (tested with ibmcloud) 
K8S_API: Needed for oc cluster only
K8S_NAME_SPACE: namespace to install models
DPM360_SERVICE_BUILDER_HOST_NAME: hostname used to create serve endpoint.
MLFLOW_TRACKING_URI: MLflow URI
MLFLOW_S3_ENDPOINT_URL: minio endpoint 
AWS_SECRET_ACCESS_KEY: minio secret (if minio is used). Default is already set
AWS_ACCESS_KEY_ID:  minio access key (if minio is used). Default is already set
