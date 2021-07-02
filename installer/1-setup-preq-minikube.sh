#!/bin/bash

# This script should install prerequisites (including minikube)

## Assumption: RHEL with root user
## Data is in /home/dpm360/data


# Usage: ./minio-upload <project_root> <minikube_ncpu>
project_root=$1
minikube_ncpu=$2

# Create user
useradd dpm360
passwd  dpm360

# To add user to sudo
usermod -aG wheel dpm360

# Add dpm360 to sudoer
echo "dpm360 ALL=(ALL) NOPASSWD: /bin/podman" >>  /etc/sudoers


# Install kubectl
curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl

## For non-root user
sudo mv ./kubectl /usr/local/kubectl

# Install minikube
su dpm360
export PATH=/usr/local/bin:$PATH

cd ~
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install podman
sudo dnf -y --refresh install podman

# Start minikube
minikube start  --driver=podman  --cpus ${minikube_ncpu}
minikube addons enable ingress

# Append ip to the /etc/hosts
echo `minikube ip` ohdsi.hcls-ibm.localmachine.io |  sudo tee -a  /etc/hosts

# Install helm charts. reference: https://helm.sh/docs/intro/install/
dnf install -y  tar
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh


## Setting up MLFlow stack services in minikube
echo "installing MLFlow using helm charts"
cd installer/model-register
minikube kubectl -- apply -f  ./model-registry/mflow-minio-db-pvc.yaml
minikube kubectl -- apply -f  ./model-registry/mflow-pg-db-pvc.yaml
minikube kubectl -- apply -f  ./model-registry/ohdsi-pg-claim-pvc.yaml

helm install modelregistry ./model-registry -n ohdsi --values ./model-registry/values.yaml
#helm uninstall modelregistry -n ohdsi

minikube dashboard --url &
echo "To look at the dashboard, use this command: minikube dashboard --url"
echo "hint: use ssh -L 30000:localhost:<dashboard_port> root@<instance_ip> to access from your machine. Also check security policies if blocked "

# switch to root user
exit
## Adding reverse proxy if you wish to access minikube outside current machine
dnf install -y nginx

echo "Update /etc/nginx/nginx.conf with following"
echo " location / { \n    proxy_pass http://ohdsi.hcls-ibm.localmachine.io/; \n }"

read -p "Press a key once ngnix.conf is update. ngnix will restart with new proxy_pass policy" -n1 -s

systemctl restart nginx
