kubectl config current-context

## Setting up PVC and namespace
kubectl apply -f ohdsi-stack/ohdsi-db-pvc.yaml
kubectl apply -f  ./model-registry/mflow-minio-db-pvc.yaml
kubectl apply -f  ./model-registry/mflow-pg-db-pvc.yaml
kubectl apply -f  ./model-registry/ohdsi-pg-claim-pvc.yaml


# Setup helm chart for model registry, object store and db (for model regitry)
helm install modelregistry ./model-registry -n ohdsi --values ./model-registry/values.yaml
# helm uninstall modelregistry -n ohdsi

# Setup helm chart for OHDSI (atlas UI, webAPI and database)
helm install  ohdsi  chgl/ohdsi -n ohdsi --values ohdsi-stack/values.yaml
#helm delete  ohdsi -n ohdsi

# Enable ingress
kubectl apply -f  ./model-registry/ingress.yaml
kubectl apply -f  ./ohdsi-stack/ingress.yaml


# Start service builder job
kubectl apply -f service-builder/cronjob.yaml -n ohdsi


## To access db
#PG_POD=$(kubectl get pod -l app=my-app -o jsonpath="{.items[0].metadata.name}")
# kubectl port-forward pods/ohdsi-postgresql-0 5432:5432 -n ohdsi

# mlflow ohdsi db
# kubectl port-forward pods/<pod name> 5431:5432 -n ohdsi