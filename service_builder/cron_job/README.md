 
 # Building the docker for the cron job
 1. Run the following command
 
     ``` 
     docker  build  -t dpm360-cronjob  .
    
     ```

 2. Tag the image and push to registry
    NB: I am using my public git account: ibmcom
    
     ``` 
     docker  tag dpm360-cronjob ibmcom/dpm360-cronjob
    
     ```
 3. Push the image   
  
     ``` 
     docker  push ibmcom/dpm360-cronjob
    
     ```
 4. After pushing the image, login to openshift or local cluster and run the following commands to start the cron  job
 
     ``` 
     oc apply -f cron_job/cronjob.yaml
     
     # or this
    
     kubectl apply -f cron_job/cronjob.yaml 
    
     ```
 5. The cron job runs after every 1 min checking for new models in mlflow and deploy as a pod
     NB: The cron job needs the following as env variables, the values here are place holders
     ```
        - name: MLFLOW_API
            value: <YOUR_VALUE>
         - name: K8S_API
            value: <YOUR_VALUE>
         - name: TOKEN
            value: <YOUR_VALUE>
     ```   
 6. To test image locally run the following command
     ```
     docker run -e MLFLOW_API=<YOUR_VALUE> -e K8S_API=<YOUR_VALUE> -e TOKEN=<> -it dpm360-cronjob

     ```   
# Testing with the script
1. To test with the python script directly and not as docker image, run the following command

   NB: Ensure that you have added the needed values:  MLFLOW_API,K8S_API & TOKEN in the env
   ```
   python model_check_create.py
   ```
2. Step 1 above will generate a deployment file and you can proceed to mannualy deploy this to k8s cluster

   NB: Please replace token with and url e.g http://127.0.0.1:56554 
     
   ```
   curl -X POST -d @deployment_json_updated.json -H "Content-Type: application/json, Authorization: Bearer <token>"  <url>/apis/apps/v1/namespaces/pmai/deployments
  
   ``` 
 
 
