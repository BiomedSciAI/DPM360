FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common wget sed build-essential git libssl-dev python3-pip jq curl ruby
RUN apt-get install -y -qq apt-transport-https ca-certificates curl
RUN apt-get install -y -qq --no-install-recommends
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic test"
RUN apt-get update && apt-get install -y docker-ce

RUN curl -sL https://ibm.biz/idt-installer | bash
RUN curl -LO https://mirror.openshift.com/pub/openshift-v4/clients/ocp/4.5.0/openshift-client-linux-4.5.0.tar.gz
RUN tar -xvf openshift-client-linux-4.5.0.tar.gz
RUN chmod +x ./oc
RUN mv ./oc /usr/local/bin/oc
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl


COPY execute.sh /
COPY deployment_template.yaml /
RUN chmod +x /execute.sh

ENTRYPOINT /bin/bash /execute.sh