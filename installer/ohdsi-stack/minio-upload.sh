#!/bin/bash

# Usage: ./minio-upload ~/data
data_folder=$1

## this is hard coded as of now ; TODO: Remove later
export MINIO_ROOT_USER=minioRoot
export MINIO_ROOT_PASSWORD=minioRoot123


export MINIO_HOST=`minikube service --url minio-mlflow-db-service -n ohdsi`

echo ${MINIO_HOST}

## install minio utiil
## TODO: verify
if !  mc --help &> /dev/null
then
    echo "mc could not be found in the path, time to install it"

    curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o $HOME/minio-binaries/mc

    chmod +x $HOME/minio-binaries/mc
    export PATH=$PATH:$HOME/minio-binaries/

    mc --help

fi

mc alias set minio-dpm360 ${MINIO_HOST}  ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} --api S3v4

mc mb minio-dpm360/1ksync
mc cp "${data_folder}"/synpuf1k.tar.gz  minio-dpm360/1ksync
mc cp "${data_folder}"/vocabulary-windows.tar  minio-dpm360/1ksync
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
mc policy get minio-dpm360/1ksync
mc policy set public minio-dpm360/1ksync
