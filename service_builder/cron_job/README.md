 
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
 