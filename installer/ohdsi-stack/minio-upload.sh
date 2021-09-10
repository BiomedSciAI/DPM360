#!/bin/bash

# Usage: ./minio-upload <username> <password> <data_folder> 
# eg. ./minio-upload minioRoot minioRoot123 ~/Downloads
MINIO_ROOT_USER=$1
MINIO_ROOT_PASSWORD=$2
data_folder=$3

export MINIO_HOST="https://`kubectl get ingress -n ohdsi minio -o=jsonpath='{.spec.rules..host}'`"

echo ${MINIO_HOST}

## install minio utiil on Linux
## TODO: verify
# if !  mc --help &> /dev/null
# then
#     echo "mc could not be found in the path, time to install it"

#     curl https://dl.min.io/client/mc/release/linux-amd64/mc \
#       --create-dirs \
#       -o $HOME/minio-binaries/mc

#     chmod +x $HOME/minio-binaries/mc
#     export PATH=$PATH:$HOME/minio-binaries/

#     mc --help

# fi

# install minio client on mac
#go get github.com/minio/mc
brew install minio/stable/mc
#mc --help

mc alias set minio-dpm360 ${MINIO_HOST}  ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} --api S3v4

# mlflow-experiments will allow mlflow to work as expected
mc mb minio-dpm360/mlflow-experiments
mc policy set readwrite minio-dpm360/mlflow-experiments

echo "Not uploading data to minio for now, we continue to use COS"
exit
# Upload data
mc mb minio-dpm360/1ksync
mc cp "${data_folder}"/synpuf1k.tar.gz  minio-dpm360/1ksync
mc cp "${data_folder}"/vocabulary-windows.tar.gz  minio-dpm360/1ksync

mc ls --recursive

echo "verify location"
mc ls --recursive minio-dpm360/1ksync/synpuf1k.tar.gz
mc ls --recursive minio-dpm360/1ksync/

mc tree minio-dpm360

mc admin info minio-dpm360

#download link
#mc share download --recursive  minio-dpm360 1ksync/

## Setting them public for now (it can anyways be accessed within the cluster because no ingress is installed
## TODO: cdm init to use boto3 to download with access / secret key: 
## Related Issue: https://github.ibm.com/IBM-Research-AI/dpm360/issues/47
#mc policy get minio-dpm360/1ksync
#mc policy set public minio-dpm360/1ksync
