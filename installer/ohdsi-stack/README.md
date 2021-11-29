# DPM360 - OHDSI stack installer

This chart is an adaptation of chart listed by [chgl/ohdsi](https://github.com/chgl/charts/tree/master/charts/ohdsi)

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
Sample value for RESULTS_TABLE_URL: "http://<webapi-service>/WebAPI/ddl/results?dialect=postgresql&schema=results&vocabSchema=cdm&tempSchema=temp&initConceptHierarchy=true"   
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

The following table lists the configurable parameters of the `ohdsi` chart and their default values.


| Parameter | Description | Default |
| - | - | - |
| imagePullSecrets | image pull secrets used by all pods | `[]` |
| nameOverride | partially override the release name | `""` |
| fullnameOverride | fully override the release name | `""` |
| commonAnnotations | annotations applied to all deployments and jobs | `[]` |
| postgresql.enabled | enable an included PostgreSQL DB. if set to`false`, the values under `webApi.db` are used | `true` |
| postgresql.postgresqlDatabase | name of the database to create see:[https://github.com/bitnami/bitnami-docker-postgresql/blob/master/README.md#creating-a-database-on-first-run](https://github.com/bitnami/bitnami-docker-postgresql/blob/master/README.md#creating-a-database-on-first-run) | `"ohdsi"` |
| postgresql.existingSecret | Name of existing secret to use for PostgreSQL passwords. The secret has to contain the keys`postgresql-password` which is the password for `postgresqlUsername` when it is different of `postgres`, `postgresql-postgres-password` which will override `postgresqlPassword`, `postgresql-replication-password` which will override `replication.password` and `postgresql-ldap-password` which will be sed to authenticate on LDAP. The value is evaluated as a template. | `""` |
| postgresql.replication.enabled | should be true for production use | `false` |
| postgresql.replication.readReplicas | number of read replicas | `2` |
| postgresql.replication.synchronousCommit | set synchronous commit mode: on, off, remote_apply, remote_write and local | `"on"` |
| postgresql.replication.numSynchronousReplicas | from the number of`readReplicas` defined above, set the number of those that will have synchronous replication | `1` |
| postgresql.metrics.enabled | should also be true for production use | `false` |
| webApi.enabled | enable the OHDSI WebAPI deployment | `true` |
| webApi.replicaCount | number of pod replicas for the WebAPI | `1` |
| webApi.db.host | database hostname | `"host.example.com"` |
| webApi.db.port | port used to connect to the postgres DB | `5432` |
| webApi.db.database | name of the database inside. If postgresql.enabled=true, then postgresql.postgresqlDatabase is used | `"ohdsi"` |
| webApi.db.username | username used to connect to the DB. Note that this name is currently used even if postgresql.enabled=true | `"postgres"` |
| webApi.db.password | the database password. Only used if postgresql.enabled=false, otherwise the secret created by the postgresql chart is used | `"postgres"` |
| webApi.db.existingSecret | name of an existing secret containing the password to the DB. | `""` |
| webApi.db.existingSecretKey | name of the key in`webApi.db.existingSecret` to use as the password to the DB. | `"postgresql-postgres-password"` |
| webApi.db.schema | schema used for the WebAPI's tables. Also referred to as the "OHDSI schema" | `"ohdsi"` |
| webApi.podAnnotations | annotations applied to the pod | `{}` |
| webApi.cors.enabled | whether CORS is enabled for the WebAPI. Sets the`security.cors.enabled` property. | `false` |
| webApi.cors.allowedOrigin | value of the`Access-Control-Allow-Origin` header. Sets the `security.origin` property. set to `*` to allow requests from all origins. if `cors.enabled=true`, `cors.allowedOrigin=""` and `ingress.enabled=true`, then `ingress.hosts[0].host` is used. | `""` |
| webApi.podSecurityContext | security context for the pod | `{}` |
| webApi.service | the service used to expose the WebAPI web port | `{"port":8080,"type":"ClusterIP"}` |
| webApi.resources | resource requests and limits for the container.<br/> 2Gi+ of RAM are recommended ([https://github.com/OHDSI/WebAPI/issues/1811#issuecomment-792988811](https://github.com/OHDSI/WebAPI/issues/1811#issuecomment-792988811)) <br/> You might also want to use `webApi.extraEnv` to set `MinRAMPercentage` and `MaxRAMPercentage`: <br/> Example: <br/> `helm template charts/ohdsi \` <br/> `--set webApi.extraEnv[0].name="JAVA_OPTS" \` <br/> `--set webApi.extraEnv[0].value="-XX:MinRAMPercentage=60.0 -XX:MaxRAMPercentage=80.0"` | `{}` |
| webApi.nodeSelector | node labels for pods assignment see:[https://kubernetes.io/docs/user-guide/node-selection/](https://kubernetes.io/docs/user-guide/node-selection/) | `{}` |
| webApi.tolerations | tolerations for pods assignment see:[https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/](https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/) | `[]` |
| webApi.affinity | affinity for pods assignment see:[https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#affinity-and-anti-affinity](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#affinity-and-anti-affinity) | `{}` |
| webApi.extraEnv | extra environment variables | `[]` |
| atlas.enabled | enable the OHDSI Atlas deployment | `true` |
| atlas.replicaCount | number of replicas | `1` |
| atlas.webApiUrl | the base URL of the OHDSI WebAPI, e.g.[https://example.com/WebAPI](https://example.com/WebAPI) if this value is not set but `ingress.enabled=true` and `constructWebApiUrlFromIngress=true`, then this URL is constructed from `ingress` | `""` |
| atlas.constructWebApiUrlFromIngress | if enabled, sets the WebAPI URL to`http://ingress.hosts[0]/WebAPI` | `true` |
| atlas.podAnnotations | annotations for the pod | `{}` |
| atlas.podSecurityContext | security context for the pod | `{}` |
| atlas.service | the service used to expose the Atlas web port | `{"port":8080,"type":"ClusterIP"}` |
| atlas.resources | resource requests and limits for the container | `{}` |
| atlas.nodeSelector | node labels for pods assignment see:[https://kubernetes.io/docs/user-guide/node-selection/](https://kubernetes.io/docs/user-guide/node-selection/) | `{}` |
| atlas.tolerations | tolerations for pods assignment see:[https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/](https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/) | `[]` |
| atlas.affinity | affinity for pods assignment see:[https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#affinity-and-anti-affinity](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#affinity-and-anti-affinity) | `{}` |
| atlas.extraEnv | extra environment variables | `[]` |
| atlas.config.local | this value is expected to contain the config-local.js contents | `""` |
| cdmInitJob.enabled | if enabled, create a Kubernetes Job running the specified container see[cdm-init-job.yaml](https://github.com/chgl/charts/blob/master/charts/ohdsi/templates/cdm-init-job.yaml) for the env vars that are passed by default | `false` |
| cdmInitJob.image | the container image used to create the CDM initialization job | `{"pullPolicy":"Always","registry":"docker.io","repository":"docker/whalesay","tag":"latest"}` |
| cdmInitJob.podAnnotations | annotations set on the cdm-init pod | `{}` |
| cdmInitJob.podSecurityContext | PodSecurityContext for the cdm-init pod | `{}` |
| cdmInitJob.securityContext | ContainerSecurityContext for the cdm-init container | `{}` |
| cdmInitJob.extraEnv | extra environment variables to set | `[]` |
| achilles.enabled | whether or not to enable the Achilles cron job | `true` |
| achilles.schedule | when to run the Achilles job. See[https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/#cron-schedule-syntax](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/#cron-schedule-syntax) | `"@daily"` |
| achilles.schemas.cdm | name of the schema containing the OMOP CDM. Equivalent to the Achilles`ACHILLES_CDM_SCHEMA` env var. | `"synpuf_cdm"` |
| achilles.schemas.vocab | name of the schema containing the vocabulary. Equivalent to the Achilles`ACHILLES_VOCAB_SCHEMA` env var. | `"synpuf_vocab"` |
| achilles.schemas.res | name of the schema containing the cohort generation results. Equivalent to the Achilles`ACHILLES_RES_SCHEMA` env var. | `"synpuf_results"` |
| achilles.cdmVersion | version of the CDM. Equivalent to the Achilles`ACHILLES_CDM_VERSION` env var. | `"5.3.1"` |
| achilles.sourceName | the CDM source name. Equivalent to the Achilles`ACHILLES_SOURCE` env var. | `"synpuf-5.3.1"` |
| ingress.enabled | whether to create an Ingress to expose the Atlas web interface | `false` |
| ingress.annotations | provide any additional annotations which may be required. Evaluated as a template. | `{}` |
| ingress.tls | ingress TLS config | `[]` |
| `CDM_URL` | Location of Athena Vocabulary file in tar.gz format.  It could be either a s3 url or a local file |   |
| `SYNPUF1K_URL` | Location of Synthetic 1K data file in tar.gz format.  It could be either a s3 url or a local file.  You can download this from[here](https://caruscloud.uniklinikum-dresden.de/index.php/s/teddxwwa2JipbXH/download). |   |
| `RESULTS_TABLE_URL` | This will be the URL to get the Results schema.<br />Example:<br />http://[server:port](server:port)/WebAPI/ddl/results?dialect=<your_cdm_database_dialect>&schema=<your_results_schema>&vocabSchema=<your_vocab_schema>&tempSchema=<your_temp_schema>&initConceptHierarchy=true |   |
| `CDM_SCHEMA` | Value of the CDM_SCHEMA in your CDM Database |   |
| `OHDSI_WEBAPI_SCHEMA` | Value of the WebAPI Schema in your database |   |
| `RESULTS_SCHEMA` | Value of Results Schema in your daabase |   |
| `TEMP_SCHEMA` | Value of Temp schema in your database |   |
|   |   |   |
|   |   |   |

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
