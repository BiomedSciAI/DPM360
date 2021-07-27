# Login specific to your cloud provider
ibmcloud login -r us-south -g '<your resource group>'
ibmcloud ks cluster config --cluster <cluster id>
# or use KUBECONFIG
#export KUBECONFIG=<path to yml file>
