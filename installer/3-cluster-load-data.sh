# Create bucket and upload training data to minio or your COS bucket
# TODO: Change the username / password combination. 3rd argument (data folder not required if not uploading datafile)
sh ohdsi-stack/minio-upload.sh minioRoot minioRoot123 ~/Downloads

## TO load data
helm install ohdsi-synpuf1k-etl chgl/ohdsi -n ohdsi --values ohdsi-stack/synpuf1k-etl.yaml
# To uninstall ETL process
# helm uninstall   ohdsi-synpuf1k-etl  -n ohdsi

## Run achillies on demand
#kubectl create job --from=cronjob/ohdsi-achilles-cron achilles-job-$(date +%s) -n ohdsi
