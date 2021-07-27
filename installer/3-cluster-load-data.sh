## TO load data
helm install ohdsi-synpuf1k-etl chgl/ohdsi -n ohdsi --values ohdsi-stack/synpuf1k-etl.yaml
# To uninstall ETL process
# helm uninstall   ohdsi-synpuf1k-etl  -n ohdsi