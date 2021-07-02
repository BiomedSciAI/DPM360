FROM ubuntu:latest

RUN apt-get update && apt-get install -y wget sed build-essential git libssl-dev python3-pip

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y software-properties-common screen git libcurl3-dev libxml2-dev r-cran-rjava openjdk-11-jdk libssl-dev libsasl2-dev libssh-dev libfontconfig1-dev libcairo2-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev librsvg2-dev
RUN apt-get install -y r-base postgresql postgresql-contrib libpostgresql-jdbc-java libpostgresql-jdbc-java-doc
RUN R CMD javareconf

RUN R -e "install.packages('drat')"
RUN R -e "install.packages('devtools')"
RUN R -e "devtools::install_github('OHDSI/FeatureExtraction')"
RUN R -e "devtools::install_github('OHDSI/PatientLevelPrediction')"

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
ENV LD_LIBRARY_PATH=/usr/lib/jvm/default-java/lib/:/usr/lib/jvm/default-java/lib/server

COPY ../service_builder/requirements.txt ./
COPY ../service_builder/ModelWrapperApp.py ./

# Copy files
COPY ../lightsaber/lightsaber ./lightsaber

RUN pip3 install -r ./requirements.txt

ENTRYPOINT ["python3", "ModelWrapperApp.py"]
