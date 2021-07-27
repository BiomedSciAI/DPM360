#!/bin/bash

echo -e  '\t' $(date) "EXPORT NEEDED VARIABLES"
echo -e  '\t' $(date) "MLFLOW_API: $MLFLOW_API"
echo -e  '\t' $(date) "K8S_CLUSTER: $K8S_CLUSTER"
echo -e  '\t' $(date) "K8S_API: $K8S_API"
echo -e  '\t' $(date) "K8S_NAME_SPACE: $K8S_NAME_SPACE"
echo -e  '\t' $(date) "DPM360_SERVICE_BUILDER_HOST_NAME: $DPM360_SERVICE_BUILDER_HOST_NAME"
echo -e  '\t' $(date) "MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo -e  '\t' $(date) "MLFLOW_S3_ENDPOINT_URL: $MLFLOW_S3_ENDPOINT_URL"
echo -e  '\t' $(date) "AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY"
echo -e  '\t' $(date) "AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID"

LOW="9001"
HIGH="9999"
export PORT=$((RANDOM * ($HIGH - $LOW + 1) / 32768 + LOW))
echo -e  '\t' $(date) $PORT

# This is appended with '' due to K8S error with numbers
export PORT_NUMBER=$(echo "'"$PORT"'")
export MLFLOW_API=$(echo $MLFLOW_API)

echo -e  '\t' $(date) "GET REGISTERED MODELS FROM MLFLOW_API"

content=$(curl -s -H "Accept: application/json" $MLFLOW_API"/api/2.0/preview/mlflow/registered-models/list")


registered_models=$(echo $content | jq ".registered_models")

echo -e  '\t' $(date) "REGISTERED MODELS" $registered_models


deploy_model() {
  # get model data
  # TODO: create a string from the returned json from mlflow
#  export MODEL_METADATA=$(curl -s -H "Accept: application/json" $MLFLOW_API"/api/2.0/mlflow/runs/get?run_id="$MODEL_RUN_ID)

  echo -e  '\t' $(date) "Compiling templates for the following moodel" $MODEL_NAME

  erb deployment_template.yaml >deployment_template_updated.yaml

  echo $(date) "$(<deployment_template_updated.yaml)"

  # Llogin to IBM cloud
  echo -e  '\t' $(date) "Login to IBM Cloud using apikey"

  ibmcloud config --check-version=false
  ibmcloud login -a cloud.ibm.com -r us-east --apikey $K8S_API_KEY
  echo -e  '\t' $(date) "Login to K8S using apikey "
  # init OC
  ibmcloud oc init
  echo $(date) "Login to K8S using apikey"
  oc login -u apikey -p $K8S_API_KEY --server=$K8S_API

  echo -e  '\t' $(date) "choose namespace"
  oc project $K8S_NAME_SPACE
  if [ $? -ne 0 ]; then
    echo -e  '\t' $(date) "Failed to authenticate to IBM Cloud"
    #    exit 1
  fi

  echo -e  '\t' $(date) "Moving on to deployment to "
  oc create -f deployment_template_updated.yaml

  echo -e  '\t' $(date) Create service

  erb service_template.yaml >service_template_updated.yaml

  echo $(date) "$(<service_template_updated.yaml)"

  oc create -f service_template_updated.yaml

  echo -e  '\t' $(date) Create ingress

  erb ingress_template.yaml >ingress_template_updated.yaml

  echo $(date) "$(<ingress_template_updated.yaml)"

  oc create -f ingress_template_updated.yaml

  echo -e  '\t' $(date) "MODEL NAME,VERSION,ROUTE: " $MODEL_NAME-$MODEL_VERSION-route
  # get model endpoint
  export model_endpoint=$(echo "http://"$DPM360_SERVICE_BUILDER_HOST_NAME/$MODEL_NAME-$MODEL_VERSION"/api")

  echo -e  '\t' $(date) "MODEL ENDPOINT " $model_endpoint

  # update mlflow
  setDeployedTagResponse=$(curl -s -H "Accept: application/json" -d '{"name":"'$MODEL_NAME_ASIS'", "key":"deployed","value":"true"}' $MLFLOW_API"/api/2.0/preview/mlflow/registered-models/set-tag")
  setDeployedVersionTagResponse=$(curl -s -H "Accept: application/json" -d '{"name":"'$MODEL_NAME_ASIS'", "key":"deployed version","value":"'$MODEL_VERSION_NUMBER'"}' $MLFLOW_API"/api/2.0/preview/mlflow/registered-models/set-tag")
  setDeployedInVersionTagResponse=$(curl -s -H "Accept: application/json" -d '{"version":"'$MODEL_VERSION_NUMBER'","name":"'$MODEL_NAME_ASIS'", "key":"deployed","value":"true"}' $MLFLOW_API"/api/2.0/preview/mlflow/model-versions/set-tag")
  setModelEndpointTagResponse=$(curl -s -H "Accept: application/json" -d '{"version":"'$MODEL_VERSION_NUMBER'","name":"'$MODEL_NAME_ASIS'", "key":"model_endpoint","value":"'$model_endpoint'"}' $MLFLOW_API"/api/2.0/preview/mlflow/model-versions/set-tag")

  # response from mlflow
  echo -e  '\t' $(date) "DEPLOYMENT OF MODEL COMPLETED " $setDeployedTagResponse $setModelEndpointTagResponse

}

for item in $(jq '.registered_models | keys | .[]' <<<"$content"); do
  item_value=$(jq -r ".registered_models[$item]" <<<"$content")
  for revisions in $(jq '.latest_versions | keys | .[]' <<<"$item_value"); do
    current_stage=$(jq -r ".latest_versions[$revisions].current_stage" <<<"$item_value")
    current_item=$(jq -r ".latest_versions[$revisions]" <<<"$item_value")
    export MODEL_SOURCE=$(jq -r ".latest_versions[$revisions].source" <<<"$item_value")
    export MODEL_RUN_ID=$(jq -r ".latest_versions[$revisions].run_id" <<<"$item_value")
    export MODEL_VERSION_NUMBER=$(jq -r ".latest_versions[$revisions].version" <<<"$item_value")
    export MODEL_VERSION=$(echo "version-"$MODEL_VERSION_NUMBER"")
    export MODEL_NAME=$(jq -r ".latest_versions[$revisions].name" <<<"$item_value" | awk '{print tolower($0)}')
    export MODEL_NAME=${MODEL_NAME//_/-}
    export MODEL_NAME_ASIS=$(jq -r ".latest_versions[$revisions].name" <<<"$item_value")
    echo -e  '\t' $(date) "CHECKING THE STATE OF THIS MODEL" $MODEL_NAME $MODEL_VERSION $current_stage $MODEL_SOURCE
    if [[ $current_item == *"tags"* ]]; then
      echo -e  '\t' $(date) "WE HAVE SOME TAGS"

      all_tags=$(jq -r ".tags" <<<"$current_item")
      echo $(date) $all_tags
      echo $(jq '.tags' <<<"$current_item")
      if [[ $all_tags == *"deployed"* ]] && [[ $all_tags == *"model_endpoint"* ]]; then
        echo -e  '\t' $(date) "MODEL ALREADY DEPLOYED"
      else
        echo -e  '\t' $(date) "MODEL NOT DEPLOYED, SO GOING TO TRY DEPLOY THIS MODEL"
        if [[ $current_stage == *"Production"* ]]; then
          echo -e  '\t' $(date) "YEEY! MODEL IN PRODUCTION SO DOING TO DEPLOY THIS, GET READY TO TRY THE MODEL IN A BIT" $MODEL_NAME $MODEL_VERSION
          deploy_model
        else
          echo -e  '\t' $(date) "OOPS, THIS MODEL IS NOT YET IN PRODUCTION. WE WILL CHECK AGAIN LATER" $MODEL_NAME $MODEL_VERSION
        fi
      fi
    else
      echo -e  '\t' $(date) "THIS MODEL HAS NO TAGS, SO GOING TO TRY DEPLOY THIS MODEL"
      if [[ $current_stage == *"Production"* ]]; then
        echo -e  '\t' $(date) "YEEY! MODEL IN PRODUCTION SO DOING TO DEPLOY THIS, GET READY TO TRY THE MODEL IN A BIT" $MODEL_NAME $MODEL_VERSION
        deploy_model
      else
        echo -e  '\t' $(date) "OOPS, THIS MODEL IS NOT YET IN PRODUCTION. WE WILL CHECK AGAIN LATER" $MODEL_NAME $MODEL_VERSION
      fi

    fi
  done
done
