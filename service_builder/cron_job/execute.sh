#!/bin/bash

echo "--------------------------------------------------------------"
echo -e '\t\t\t' "EXPORT NEEDED VARIABLES"
echo "--------------------------------------------------------------"

echo "MLFLOW_API: $MLFLOW_API"
echo "OPENSHIFT_CLUSTER: $OPENSHIFT_CLUSTER"
echo "OPENSHIFT_API_KEY: $OPENSHIFT_API_KEY"
echo "OPENSHIFT_API: $OPENSHIFT_API"
echo "OPENSHIFT_PROJECT: $OPENSHIFT_PROJECT"

echo "--------------------------------------------------------------"
echo -e '\t\t\t' "GET REGISTERED MODELS FROM MLFLOW_API"
echo "--------------------------------------------------------------"

content=$(curl -s -H "Accept: application/json" $MLFLOW_API"/api/2.0/preview/mlflow/registered-models/list")

registered_models=`echo $content | jq ".registered_models" `


deploy_model()
{
 
  echo "--------------------------------------------------------------"
  echo "Compiling the manifest template for the following environment"
  echo "--------------------------------------------------------------"
  echo -e '\t\t\t' $MODEL_NAME
  echo "--------------------------------------------------------------"
  
  erb deployment_template.yaml  > deployment_template_updated.yaml
  
  echo "--------------------------------------------------------------"
  echo "$(<deployment_template_updated.yaml)"
  echo -e '\t\t\t' "Done compiling manifest"
  echo "--------------------------------------------------------------"

  # init OC
  ibmcloud oc init

  # Llogin to IBM cloud
  echo "Login to IBM Cloud using apikey"
  ibmcloud login -a cloud.ibm.com -r us-east --apikey $OPENSHIFT_API_KEY
  oc login -u apikey -p $OPENSHIFT_API_KEY --server=$OPENSHIFT_API
  oc project $OPENSHIFT_PROJECT
  if [ $? -ne 0 ]; then
    echo "Failed to authenticate to IBM Cloud"
    exit 1
  fi
  
  echo "--------------------------------------------------------------"
  echo -e '\t\t\t' "Moving on to deployment to " $OPENSHIFT_CLUSTER $OPENSHIFT_API $OPENSHIFT_PROJECT
  echo "--------------------------------------------------------------"
  oc create -f deployment_template_updated.yaml

  # expose the model
  oc expose deploy $MODEL_NAME-$MODEL_VERSION --name $MODEL_NAME-$MODEL_VERSION
  oc expose service $MODEL_NAME-$MODEL_VERSION

  # get model endpoint
  export model_endpoint=`echo "$(oc  get route $MODEL_NAME-$MODEL_VERSION  --template='http://{{.spec.host}}')" `

  echo $model_endpoint

  # update mlflow
  setDeployedTagResponse=$(curl -s -H "Accept: application/json" -d '{"name":"'$MODEL_NAME_ASIS'", "key":"deployed","value":"true"}' $MLFLOW_API"/api/2.0/preview/mlflow/registered-models/set-tag")

  setModelEndpointTagResponse=$(curl -s -H "Accept: application/json" -d '{"name":"'$MODEL_NAME_ASIS'", "key":"model_endpoint","value":"'$model_endpoint'"}' $MLFLOW_API"/api/2.0/preview/mlflow/registered-models/set-tag")

  # response from mlflow
  echo $setDeployedTagResponse
  echo $setModelEndpointTagResponse
  echo "--------------------------EXECUTION STARTED------------------------------------"
  echo -e '\t\t\t' "DEPLOYMENT OF MODEL COMPLETED " $setDeployedTagResponse $setModelEndpointTagResponse $model_endpoint
  echo "---------------------------EXECUTION STARTED-----------------------------------"


}
echo "--------------------------DONE------------------------------------"


for item in $(jq '.registered_models | keys | .[]' <<< "$content");
do
    item_value=$(jq -r ".registered_models[$item]" <<< "$content");

    # We check if we have tags
    if [[ $item_value == *"tags"* ]];
    then
      echo "It's there!"
       for tags in $(jq '.tags | keys | .[]' <<< "$item_value");
      do
        tag_key=$(jq -r ".tags[$tags].key" <<< "$item_value");
        tag_value=$(jq -r ".tags[$tags].value" <<< "$item_value");
        echo ${tag_key}
        if [[ $tag_key == *"deployed"*  ]] && [[ $tag_value == *"true"* ]];then
          echo "---------- MODEL ALREADY DEPLOYED ------"
        else
          echo "---------- MODEL NOT DEPLOYED WE CHECK IF IN PRODUCTION------"
          for revisions in $(jq '.latest_versions | keys | .[]' <<< "$item_value");
          do
            current_stage=$(jq -r ".latest_versions[$revisions].current_stage" <<< "$item_value");
            echo ${current_stage}
            if [[ $current_stage == *"Production"* ]];then
              echo "----------FOUND  MODEL IN PRODUCTION WITH TAGS------"
              export MODEL_SOURCE=$(jq -r ".latest_versions[$revisions].source" <<< "$item_value");
              export MODEL_RUN_ID=$(jq -r ".latest_versions[$revisions].run_id" <<< "$item_value");
               export MODEL_NAME=$(jq -r ".latest_versions[$revisions].name" <<< "$item_value" | awk '{print tolower($0)}')
              export MODEL_NAME_ASIS=$(jq -r ".latest_versions[$revisions].name" <<< "$item_value");

              export MODEL_VERSION=$(jq -r ".latest_versions[$revisions].version" <<< "$item_value");
              deploy_model
            else
              echo "---------- MODEL NOT IN PRODUCTION ------"
            fi
          done
        fi
      done
    else
      for revisions in $(jq '.latest_versions | keys | .[]' <<< "$item_value");
      do
        current_stage=$(jq -r ".latest_versions[$revisions].current_stage" <<< "$item_value");

        echo ${current_stage}
        if [[ $current_stage == *"Production"* ]];then
          echo "----------FOUND  MODEL IN PRODUCTION ------"
          export MODEL_SOURCE=$(jq -r ".latest_versions[$revisions].source" <<< "$item_value") ;
          export MODEL_RUN_ID=$(jq -r ".latest_versions[$revisions].run_id" <<< "$item_value");
          export MODEL_VERSION=$(jq -r ".latest_versions[$revisions].version" <<< "$item_value");
          export MODEL_NAME=$(jq -r ".latest_versions[$revisions].name" <<< "$item_value" | awk '{print tolower($0)}')
           export MODEL_NAME_ASIS=$(jq -r ".latest_versions[$revisions].name" <<< "$item_value");

          deploy_model
        else
          echo "---------- MODEL NOT IN PRODUCTION ------"
        fi
      done
    echo ${value}
    echo ${k}
    fi
done

echo "-----------------------------EXECUTION COMPLETED---------------------------------"
