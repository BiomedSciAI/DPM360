
# Deployment of full OHDSI technology stack for a non-cluster environment

## About OHDSI Broadsea


You can use [OHDSI Broadsea](https://github.com/OHDSI/Broadsea) to build a docker container on your VM or server (hereafter, the host), which includes necessary OHDSI technologies such as ATLAS, WebAPI, Achilles, R Methods Library and others.


Refer to [README.md](https://github.com/OHDSI/Broadsea/blob/master/README.md) for general information of dependencies and installation:
- Broadsea Dependencies
  - https://github.com/OHDSI/Broadsea#broadsea-dependencies
- Quick Start Broadsea Deployment
  - https://github.com/OHDSI/Broadsea#quick-start-broadsea-deployment

## Database Setup


[Broadsea](https://github.com/OHDSI/Broadsea) supports Apache Impala, Oracle, MS SQL Server, PostgreSQL. Here we show an installation guide where you can install and run PostgreSQL on the host in which you are running docker containers.


After installing PostgreSQL in the host, create the user and database using psql commands assuming:
- username: `dpm360`
- password: `dpm360-password`
- database name: `dpm360db`


Next configure PostgreSQL to allow a docker VM to access to the PostgreSQL database. 
Please confirm the IP address of docker0 (virtual network bridge on the host) by<br>
`ip address show dev docker0`.<br>
**Note**, you can see this address before starting containers (it is usually ok if the docker service is on). Here, we assume the ip is `172.17.0.1`


Using this IP address, modify configuration files:

<pre>
/etc/postgresql/10/main/pg_hba.conf:
add the following line
host    all    all    172.17.0.1/0    md5
</pre>

<pre>
/etc/postgresql/10/main/postgresql.conf:
change listen_addresses variable as
listen_addresses = 'localhost, 172.17.0.1'
</pre>

## Broadsea Container Deployment and Run


To run docker container for OHDSI stack (ATLAS, WebAPI, and Achilles), follow instructions below. 

- change directory to &lt;dpm360 root dir&gt;/installer/express/broadsea-example
- modify docker-compose.yml for your environment

<pre>
services:
  broadsea-webtools:
    ports:
      - "18080:8080" # change host port 18080 if needed
    environment:
      - WEBAPI_URL=http://172.17.0.1:18080 # confirm address and port
      - datasource_url=jdbc:postgresql://172.17.0.1:5432/dpm360db # confirm address and postgresql configuration
      - datasource_username=dpm360 # confirm postgresql configuration
      - datasource_password=dpm360-password # confirm postgresql configuration
      - flyway_datasource_url=jdbc:postgresql://172.17.0.1:5432/dpm360db # confirm address and postgresql configuration
      - flyway_datasource_username=dpm360 # confirm postgresql configuration
      - flyway_datasource_password=dpm360-password # confirm postgresql configuration
</pre>

- run `docker-compose up -d`
  - please wait for while and access to &lt;host&gt;:18080/atlas/ to confirm it is started

## Define and Populate OHDSI OMOP-CDM Database on PostgreSQL

Next step is to setup PostgreSQL by defining tables and importing data. We provide a custom docker image to initialize the CDM database with Athena Vocabularies. Currently this only suppors [SynPUF 1k data](https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs) but we plan to release a more general tool to setup the database.


First, you make a vocabulary file.  All necessary vocabulary files can be downloaded from the ATHENA download site: http://athena.ohdsi.org. A tutorial for Athena is available at https://www.youtube.com/watch?v=2WdwBASZYLk. Download guide is given from 10:04. According to the guidance, please make vocabs.tar.gz, and put the file at:
<pre>
&lt;dpm360 root dir&gt;/installer/express/cdm-init-example-local/data/vocabs.tar.gz
</pre>

Please confirm vocabs.tar.gz includes the followings (confirm it has no directory structure):
<pre>
  CONCEPT_ANCESTOR.csv
  CONCEPT_CLASS.csv
  CONCEPT_RELATIONSHIP.csv
  CONCEPT_SYNONYM.csv
  CONCEPT.csv
  DOMAIN.csv
  DRUG_STRENGTH.csv
  RELATIONSHIP.csv
  VOCABULARY.csv
</pre>

Next, obtain SynPUF 1k (CDM 5.3.1) data from [here](https://caruscloud.uniklinikum-dresden.de/index.php/s/teddxwwa2JipbXH/download). You have to change the directory structure as expected. Try the following:

`tar -zxvf synpuf1k.tar.gz *.csv`<br>
`cd synpuf1k531`<br>
`tar -zcvf synpuf1k.tar.gz *.csv`<br>
and put the file at:
<pre>
&lt;dpm360 root dir&gt;/installer/express/cdm-init-example-local/data/synpuf1k.tar.gz
</pre>

Please confirm synpuf1k.tar.gz includes the followings (confirm it has no directory structure):
<pre>
  visit_occurrence.csv
  care_site.csv
  cdm_source.csv
  condition_era.csv
  condition_occurrence.csv
  cost.csv
  death.csv
  device_exposure.csv
  drug_era.csv
  drug_exposure.csv
  location.csv
  measurement.csv
  observation_period.csv
  observation.csv
  payer_plan_period.csv
  person.csv
  procedure_occurrence.csv
  provider.csv
</pre>

The following instructions then run a docker container to prepare the database.

- change directory to &lt;dpm360 root dir&gt;/installer/express/cdm-init-example-local
- modify docker-compose.yml for your environment

<pre>
services:
  cdmInitJob:
    image: ibmcom/dpm360-cdm_init:1.2
    volumes:
      - ./data:/data # /data is mounted to ./data of the host
    environment:
      - CDM_URL=file:///data/vocabs.tar.gz # /data is mounted to the host, confirm file name is correct
      - SYNPUF1K_URL=file:///data/synpuf1k.tar.gz # /data is mounted to the host, confirm file name is correct
</pre>

- run `docker-compose up`
  - wait untill it ends
  
## Run Achilles (Recommended but Optional)


[Achilles](https://ohdsi.github.io/Achilles/) computes statistics on your OMOP CDM database.


Follow instructions below to run a docker container to make Achilles work for your database.

- change directory to &lt;dpm360 root dir&gt;/installer/express/achilles-example
- run `docker-compose up`
  - wait untill it ends
  - access to &lt;host&gt;:18080/atlas/ and click "Data source" to see the statistics of your database

## Model Registry

You can run MLFlow on the host as Model Registry, which can be connected to lightsaber (model training framework) and service builder (micro service builder using the trained model). A guidance is being prepared.


## What To Do Next

- use Atlas &lt;host&gt;:18080/atlas/ to define cohorts and outcomes
- use [cohort tools](../../cohort_tools/docs/index.md) to extract features to make training data
- use [lightsaber](../../lightsaber/docs/index.md) to build and train the model using above data
- use **service builder** to deploy a service using the trained model
