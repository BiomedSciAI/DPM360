
# Deployment of full OHDSI technology stack for non-cluster environments

## About OHDSI Broadsea


You can use [OHDSI Broadsea](https://github.com/OHDSI/Broadsea) to build a docker container on your VM or server (hereafter, the host), which includes necessary OHDSI technologies such as ATLAS, WebAPI, Achilles, R Methods Library and others.


Please have look at README.md for general information of dependencies and installation:
- Broadsea Dependencies
  - https://github.com/OHDSI/Broadsea#broadsea-dependencies
- Quick Start Broadsea Deployment
  - https://github.com/OHDSI/Broadsea#quick-start-broadsea-deployment

## Database Setup


[Broadsea](https://github.com/OHDSI/Broadsea) supports Apache Impala, Oracle, MS SQL Server, PostgreSQL. Here we show an installation guide where you install and run PostgreSQL on the host in which you are running docker containers.


After installing PostgreSQL to the host, create the user and database using psql commands assuming:
- username: dpm360
- password: dpm360-password
- database name: dpm360db


Next configure PostgreSQL to allow a docker VM to access to the PostgreSQL database. Please confirm your IP address of docker0 (virtual network bridge on the host) by<br>
`ip address show dev docker0`.<br>
Note that you can see this address before starting containers. In this case, we assume it is 172.17.0.1


Using this IP address, modify configuration files:


**/etc/postgresql/10/main/pg_hba.conf**

add the following line:<br>
host    all    all    172.17.0.1/0    md5

**/etc/postgresql/10/main/postgresql.conf**

change listen_addresses variable as<br>
listen_addresses = 'localhost, 172.17.0.1'

