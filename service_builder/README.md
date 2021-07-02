# Service Builder
The Service Builder component automatically converts registered models in model registry into microservices. This is achieved through the use of hooks in the model registry. These hooks listen for production ready models in the registry and thereafter starts model packaging execution pipeline.  The pipeline include extraction of model metadata(e.g. model artifacts, feature generation files, data, etc) from registry, model containerization, authentication, installaction and deployment in the target cluster(Kubernates or OpenShift). Upon successful model deployment,  model registry callback functionality updates model metadata in the registry with deployment status and model access endpoint. Using the supplied endpoint users are provided with intuitive swagger based interface where the deployed model be accessed for serrving.  

# building service builder image
1. Run the following command
 
     ``` 
    docker  build  -t dpm360-service-builder -f service_builder/Dockerfile .
     ```

 2. Tag the image and push to registry
    NB: I am using my public git account: ibmcom
    
     ``` 
     docker  tag dpm360-service-builder ibmcom/dpm360-service-builder
    
     ```
 3. Push the image   
  
     ``` 
     docker  push ibmcom/dpm360-service-builder
    
     ```
 # Testing    
 1. To test image locally run the following command
 
    ```
    docker run --publish=0.0.0.0:8084:9090 -e MLFLOW_S3_ENDPOINT_URL=<YOUR_VALUE_HERE> -e AWS_ACCESS_KEY_ID=<YOUR_VALUE_HERE> -e AWS_SECRET_ACCESS_KEY=<YOUR_VALUE_HERE> -e MODEL_NAME=<YOUR_VALUE_HERE> -e MODEL_VERSION=<YOUR_VALUE_HERE> -e MODEL_RUN_ID=<YOUR_VALUE_HERE> -e MLFLOW_TRACKING_URI=<YOUR_VALUE_HERE> -e MODEL_SOURCE=<YOUR_VALUE_HERE> -ti dpm360-service-builder

    ``` 
 2. To test model wrapper code locally, set the following env variables 
     ``` 
    export MODEL_NAME=<YOUR_VALUE_HERE>
    export MODEL_VERSION=1
    export FEATURE_GENERATOR_FILE_NAME=<YOUR_VALUE_HERE>
    export MLFLOW_TRACKING_URI=<YOUR_VALUE_HERE><YOUR_VALUE_HERE>
    export MODEL_SOURCE=<YOUR_VALUE_HERE>
    export MODEL_RUN_ID=<YOUR_VALUE_HERE><YOUR_VALUE_HERE>
    export AWS_ACCESS_KEY_ID=<YOUR_VALUE_HERE>
    export AWS_SECRET_ACCESS_KEY=<YOUR_VALUE_HERE>
    export MLFLOW_S3_ENDPOINT_URL=<YOUR_VALUE_HERE>
    
     ```
 3. Then start flask app
    ```
    python ModelWrapperApp.py    
    
    ```
  
    To access the swagger page
    
    ```
    http://0.0.0.0:8080/doc/
    
    ```       
