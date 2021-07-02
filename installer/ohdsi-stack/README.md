# DPM360 - OHDSI stack installer

This chart is built on top of the chart listed at [chgl/ohdsi](https://github.com/chgl/charts/tree/master/charts/ohdsi)

A sample values.yaml file is provided in this repository.

## Introduction

This chart deploys the OHDSI WebAPI and ATLAS app. on a [Kubernetes](http://kubernetes.io) cluster using the [Helm](https://helm.sh) package manager.

##

## Prerequisites

* Kubernetes v1.18+
* Helm v3
* Kubernetes namespace.
* Persistent Volume claims for Postgres Database ( Refer to ohdsi-db-pvc.yaml for pvc configuration)

##

## Installing the Chart

To install the chart with the release name `ohdsi`:

```
$ helm repo add chgl https://chgl.github.io/charts
$ helm repo update

Update host value under ingress.hosts
$ helm install ohdsi chgl/ohdsi -n <your namespace> --values values.yaml


Update PGPASSWORD, CDM_URL, SYNPUF1K_URL, RESULTS_TABLE_URL values as needed from synpuf1k-etl.yaml. 
Sample value for RESULTS_TABLE_URL: "http://<webapi-service>/WebAPI/ddl/results?dialect=postgresql&schema=results&vocabSchema=cmd&tempSchema=temp&initConceptHierarchy=true"   
Upload raw data into minio bucket. Obtain URLs, CDM_URL and SYNPUF1K_URL, for files hosted in minio. 
$ helm install ohdsi-synpuf1k-etl chgl/ohdsi -n <your namespace> --values synpuf1k-etl.yaml
```

The command deploys the OHDSI WebAPI and ATLAS app. on the Kubernetes cluster in the default configuration. The [configuration](https://github.com/chgl/charts/tree/master/charts/ohdsi#configuration) section lists the parameters that can be configured during installation.

> **Tip**: List all releases using `helm list`

##

## Uninstalling the Chart

To uninstall/delete the `ohdsi`:

<pre>$ <span class="pl-s1">helm delete ohdsi -n ohdsi</span></pre>

The command removes all the Kubernetes components associated with the chart and deletes the release.

##

## Configuration

Refer to the original repository for the configuration parameters.

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`. For example:

<pre>$ <span class="pl-s1">helm install ohdsi chgl/ohdsi -n ohdsi --set postgresql.postgresqlDatabase=<span class="pl-s"><span class="pl-pds">"</span>ohdsi<span class="pl-pds">"</span></span></span></pre>

Alternatively, a YAML file that specifies the values for the parameters can be provided while
installing the chart. For example:

<pre>$ <span class="pl-s1">helm install ohdsi chgl/ohdsi -n ohdsi --values values.yaml</span></pre>

##

## Initialize the CDM using a custom container

1. A custom docker image to initialize the CDM database with Athena Vocabularies and Synthetic 1K patient data is built based on the broad guidelines outlined [here](https://github.com/IBM/DPM360).
   This custom image is utilized in the cdmInitJob.image parameter in the synpuf1k-etl.yaml.

   The cdmInit container takes in the following parameters to initialize the data:

   `CDM_URL`Location of Athena Vocabulary file in tar.gz format.  It could be either a s3 url or a local file `SYNPUF1K_URL`Location of Synthetic 1K data file in tar.gz format.  It could be either a s3 url or a local file.  You can download this from[here](https://caruscloud.uniklinikum-dresden.de/index.php/s/teddxwwa2JipbXH/download). ` RESULTS_TABLE_URL`This will be the URL to get the Results schema.
   Example:
   http://[server:port](server:port)/WebAPI/ddl/results?dialect=<your_cdm_database_dialect>&schema=<your_results_schema>&vocabSchema=<your_vocab_schema>&tempSchema=<your_temp_schema>&initConceptHierarchy=true

   `CDM_SCHEMA` Name of the schema that contains the CDM tables in your database.

   `OHDSI_WEBAPI_SCHEMA` Name of the schema that contains the WebAPI tables in your database.

   `RESULTS_SCHEMA`Name of the schema that contains the results tables in your database.
   `TEMP_SCHEMA` Name of the schema that contains the temp results table in your database.

# Troubleshooting

If the deployment does not work, possible todo following this https://www.kubeflow.org/docs/started/workstation/minikube-linux/
