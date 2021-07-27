#!/bin/bash

## Assumption:
# RHEL with user "dpm360"
# data is in /home/dpm360/data
# minikube is runnning
## code resides in ~/dpm360

cd ~/dpm360

echo "code"
ls -lah ~/dpm360


echo "Load data to minio"
## Upload data to minio
mkdir -p /home/dpm360/data

read -p "Copy vocab file and synthetic data to /home/dpm360/data. Once copied press any key to continue... " -n1 -s

ls mkdir -p /home/dpm360/data

cd ~/dpm360/installer/ohdsi-stack

./minio-upload.sh ~/data


## Setting up OHDSI services in minikube
echo "installing ODHSI using helm charts"
cd installer/ohdsi-stack

minikube kubectl -- apply -f ohdsi-db-pvc.yaml

helm repo add chgl https://chgl.github.io/charts
helm repo update

helm install  ohdsi  chgl/ohdsi -n ohdsi --values values.yaml
#helm delete  ohdsi -n ohdsi


## Starting ETL job in minikube
echo "Starting ETL process for Sythetic data"
helm install ohdsi-synpuf1k-etl chgl/ohdsi -n ohdsi --values synpuf1k-etl.yaml
#helm uninstall   ohdsi-synpuf1k-etl  -n ohdsi


## Setup ingress
cd ..
# /mlflow , /webapi , /atlas will be enabled
minikube kubectl -- apply -f  model-registry/ingress.yaml -n ohdsi
# TODO: We may want to use ingress from the helm chart
#minikube kubectl -- apply -f  ohdsi-stack/ingress.yaml -n ohdsi

## Service builder
echo "Starting service builder"
minikube kubectl -- apply -f cronjob.yaml -n ohdsi
## Service builder code is : https://github.ibm.com/IBM-Research-AI/dpm360/tree/dev_deployment_pipeline/service_builder


