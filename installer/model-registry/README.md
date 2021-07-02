# MLFlow and dependcies

[DPM360](https://github.com/ibm/dpm360) - Helm chart for deploying Disease progression model framework including OHDSI tools ( Atlas, WebAPI), MLFlow and its dependencies ( minio for object storage and postgresql for relational database).


## Pre-requisites

```console
1) Download the chart repo

2) In your container platform environment create a namespace, <your namespace>.

3) In your container platform environment create 3 persistent volume claims ( one for OHDSI postgres, one for minio-mlflow and one for postgres-mlflow). Refer to ohdsi-pg-claim-pvc.yaml, mflow-pg-db-pvc.yaml and mflow-minio-db-pvc.yaml files. Create PVCs via Minikube UI.

4) Update the values.yaml with parameters matching your container platform environment. This would include PVCs and Memory/CPU allocation to the different MLFlow components.

5) Setup access host as needed, for example ohdsi.hcls-ibm.localmachine.io under /etc/hosts for Linux

```

## Introduction

This chart deploys the MLFlow along with a minio based storage and postgesql database on a [Kubernetes](http://kubernetes.io) cluster using the [Helm](https://helm.sh) package manager.

## Prerequisites

- Kubernetes v1.18+
- Helm v3

## Installing the Chart

```console
Once you have cloned the repo (https://github.com/IBM/DPM360)

$ cd to the folder where you have installer/
$ helm install modelregistry ./model-registry -n <your namespace> --values ./model-registry/values.yaml
This will create 3 deployments in your kubernetes cluster ( mlflow, minio and postgresql)

Update your ingress to allow access to the services created by the helm chart.

```

The command deploys the MLFlow (version 1.14.1) along with minio for storage and postgresql on the Kubernetes cluster in the default configuration. The [configuration](#configuration) section lists the parameters that can be configured during installation.

> **Tip**: List all releases using `helm list`

## Uninstalling the Chart

To uninstall/delete the `modelregistry`:

```console
$ helm delete modelregistry -n <your namespace>
```

The command removes all the Kubernetes components associated with the chart and deletes the release.

## Configuration

The following table lists the configurable parameters of the `model-registry` chart and their default values.


| Parameter | Description | Default |
| - | - | - |
| MINIO | This section is for minio configuration |   |
| minio.image | minio images used for this installation | `""` |
| minio.accessKey | access key ( usename) required by minio | `""` |
| minio.secretKey | secret key required by minio | `[]` |
| minio.rootuser | minio console user | `""` |
| minio.rootpassword | minio console user password | `""` |
| minio.resources | section used to configure your pod memory and cpu settings | `""` |
| minio.persistence | This section specifies the PVC that you had created a part of the pre-requisites | `true` |
| minio.container port | container port ( typically set to 9000) | `9000` |
| minio.httpport | port that is exposed in your service specification. Typcially set to 9000 | `"9000"` |
|   |   |   |
| Postgres for MLFlow | This section describes Postgres for MLFlow configuration |   |
| pgmlflow.enabled | enable the postgres deployment for mlflow | `true` |
| pgmlflow.image | postgres images used ( 12.5 in this example) | `1` |
| pgmlflow.POSTGRES_USER | postgres user used for the installation | `"postgres"` |
| pgmlflow.POSTGRES_PASSWORD | password for the postgres user | `postgres` |
| pgmlflow.resources | use this section to specify the pod memery and cpu limits | `""` |
| pgmlflow.containerport | container port for postgres db | `"5432"` |
| pgmlflow.httpport | port for running the postgres service.  If you have multiple postgres instances, this will be different from the container port | `"5452"` |
|   |   |   |
| MLFlow | This section lists the configuration for MLFlow |   |
| mlflow.enabled | enable the mlflow for this deployment | `"true"` |
| mlflow.image | specifies the mlflow image used | `{}` |
| mlflow.MLFLOW_HOST | MLFlow host name | `` |
| mlflow.BACKEND_STORE_URI | datastore used for backend.  In our case we have used postgresql | `""` |
| mlflow.POSTGRES_HOST | postgres service name | `{}` |
| mlflow.MINIO_HOST | minio endpoint that will be exposed by the ingress | `{}` |
| mlflow.MLFLOW_TRACKING_URI | mlflow endpoit that will exposed by the ingress | `{}` |
| mlflow.MLFLOW_S3_ENDPOINT_URL | minio endpoint that will be exposed by the ingress. | `{}` |
|   |   |   |
| mlflow.AWS_ACCESS_KEY_ID | minio user id | `{}` |
| mlflow.AWS_SECRET_ACCESS_KEY | minio access key for the user | `[]` |
| mlflow.AWS_MLFLOW_BUCKET_NAME<br />mlflow.AWS_BUCKET  / AWS_MLFLOW_BUCKET | name of the bucket used for mlflow experiments<br /> | `mlflow-experiments` |
| mlflow.resources | use this section to define the memory and cpu for the pod | `1` |
| mlflow.containerport | port number of the container. Typically it is 5000 | `"9000"` |
| mlflow.httpport | port number that the service listens.  Typically same as containerport | `"9000"` |
|   |   |   |
|   |   |   |
|   |   |   |

Specify each parameter using the  YAML file that specifies the values for the parameters while installing the chart. For example:

```console
$ helm install modelregistry ./model-registry -n <your namespace> --values ./model-registry/values.yaml
```

##

```

```
