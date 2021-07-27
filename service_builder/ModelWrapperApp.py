"""
Needed dependencies
"""
import logging
import os
from pathlib import Path
import mlflow
import pandas as pd
import torch as T
from flask import Flask, request, jsonify, render_template,send_from_directory
from flask_httpauth import HTTPBasicAuth
from flask_restplus import Api, Resource, fields
from sklearn.preprocessing import StandardScaler
from lightsaber.data_utils import pt_dataset as ptd
import lightsaber.data_utils.utils as du
from mlflow.tracking import MlflowClient
import json
import shutil
import requests
log = logging.getLogger()

"""
The following two functions is what is needed by lightsaber code, we will move this to the model at a later point
"""
@ptd.functoolz.curry
def filter_fillna(data, target, fill_value=0., time_order_col=None):
    data = data.copy()

    idx_cols = data.index.names
    if time_order_col is not None:
        try:
            sort_cols = idx_cols + time_order_col
        except:
            sort_cols = idx_cols + [time_order_col]
    else:
        sort_cols = idx_cols

    data.update(data.reset_index()
                .sort_values(sort_cols)
                .groupby(idx_cols[0])
                .ffill())

    data.fillna(fill_value, inplace=True)

    return data, target
@ptd.functoolz.curry
def filter_preprocessor(data, target, cols=None, preprocessor=None, refit=False):
    if preprocessor is not None:
        all_columns = data.columns
        index = data.index

        # Extracting the columns to fit
        if cols is None:
            cols = all_columns
        _oCols = all_columns.difference(cols)
        xData = data[cols]

        # If fit required fitting it
        if refit:
            preprocessor.fit(xData)
            log.info(f'Fitting pre-proc: {preprocessor}')

        # Transforming data to be transformed
        try:
            xData = preprocessor.transform(xData)
        except NotFittedError:
            raise Exception(f"{preprocessor} not fitted. pass fitted preprocessor or set refit=True")
        xData = pd.DataFrame(columns=cols, data=xData, index=index)

        # Merging other columns if required
        if not _oCols.empty:
            tmp = pd.DataFrame(data=data[_oCols].values,
                               columns=_oCols,
                               index=index)
            xData = pd.concat((tmp, xData), axis=1)

        # Re-ordering the columns to original order
        data = xData[all_columns]
    return data, target

"""
ModeWrapperApp Flask App, which is used to fetch and serve the model
"""
class ModeWrapperApp(Flask):
  def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):

    """
    Get env variables
    """
    self.model_name = os.environ['MODEL_NAME']
    self.model_version = os.environ['MODEL_VERSION']
    self.run_id = os.environ["MODEL_RUN_ID"]
    self.model_source_uri = os.environ["MODEL_SOURCE"]

    """
    String literals
    """
    # we replace -version-1 in model name since the yaml don't come with versions
    self.model_yaml_file = self.model_name.strip().replace("-" + self.model_version, "") + ".yaml"
    self.model_yaml_file_caps_case = self.model_name.strip().replace("-" + self.model_version, "").upper() + ".yaml"
    self.model_yaml_file_small_case = self.model_name.strip().replace("-" + self.model_version, "").lower() + ".yaml"
    self.model_y_test_file_name = "y_test.csv"

    """
    # Fetch model from mlflow registry using model source uri e.g. s3://mlflow-experiments/0/cfd8c04976a04d24b9d2ded788903beb/artifacts/model
    """
    self.model = mlflow.pytorch.load_model(model_uri=self.model_source_uri)

    """
    Create mlflow client which is used to download model artifacts
    """
    self.mlflowClient =MlflowClient()

    """
    Download artifacts from mlflow
    """
    local_dir = os.getcwd()
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    local_path=self.mlflowClient.download_artifacts(self.run_id, "features", local_dir)

    """
    Confirm downloaded files
    """
    print("Artifacts downloaded in: {}".format(local_path))
    print("Artifacts: {}".format(os.listdir(local_path)))

    """
    Load model yaml from mode features directory
    """
    model_yaml_dir_path_caps = local_path + '/' + str(self.model_yaml_file_caps_case).strip()
    model_yaml_dir_path_small = local_path + '/' + str(self.model_yaml_file_small_case).strip()
    self.model_y_test_file = local_path + '/' + self.model_y_test_file_name
    self.features_local_path = local_path

    if os.path.exists(model_yaml_dir_path_caps):
        expt_conf = du.yaml.load(
            open(model_yaml_dir_path_caps).read().format(DATA_DIR=local_path),
            Loader=du._Loader)
    elif os.path.exists(model_yaml_dir_path_small):
        expt_conf = du.yaml.load(
            open(model_yaml_dir_path_small).read().format(DATA_DIR=local_path),
            Loader=du._Loader)

    """
      Load model data
    """
    preprocessor = StandardScaler()
    train_filter = [filter_preprocessor(cols=expt_conf['numerical'],
                                                        preprocessor=preprocessor,
                                                        refit=True),
                    filter_fillna(fill_value=expt_conf['normal_values'],
                                                  time_order_col=expt_conf['time_order_col'])
                    ]
    transform = ptd.transform_drop_cols(cols_to_drop=expt_conf['time_order_col'])

    self.model_dataset = ptd.BaseDataset(tgt_file=expt_conf['test']['tgt_file'],
                                        feat_file=expt_conf['test']['feat_file'],
                                        idx_col=expt_conf['idx_cols'],
                                        tgt_col=expt_conf['tgt_col'],
                                        feat_columns=expt_conf['feat_cols'],
                                        time_order_col=expt_conf['time_order_col'],
                                        category_map=expt_conf['category_map'],
                                        transform=transform,
                                        filter=train_filter,
                                        )

    super(ModeWrapperApp, self).run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)

"""
Flask App Init
"""
app = ModeWrapperApp(__name__,static_url_path="/" + os.environ['MODEL_NAME']+'/static')
api = Api(app, version='1.0', title='DPM360:'+os.environ['MODEL_NAME'], prefix="/" + os.environ['MODEL_NAME']+"/api/v1",description='The API for Disease Progression Modeling Workbench 360:'+os.environ['MODEL_NAME'], doc="/" + os.environ['MODEL_NAME']+'/doc/')

"""
Basic Auth
"""
auth = HTTPBasicAuth()

"""
 Prepare static files
"""
dir_name=str("/"+os.environ['MODEL_NAME'])
if not os.path.exists(os.getcwd() + dir_name):
    folder= os.getcwd() + dir_name
    os.umask(0)
    os.makedirs(folder,mode=0o777)
    destination= os.getcwd() + dir_name+"/static"
    shutil.copytree(os.getcwd() + "/static", destination)


"""
Update static api json with base root matching model name
"""
static_api_path = './' + str(os.environ['MODEL_NAME']+"/static/api.json")
static_api_dir = (Path.cwd() / static_api_path).resolve()
with open(static_api_dir, 'r') as fp:
    api_data = json.load(fp)
api_data["basePath"]="/" + os.environ['MODEL_NAME']+"/api/v1"
api_data["info"]["title"]= "DPM360:"+os.environ['MODEL_NAME']
# get model metadata
response = requests.get(os.environ["MODEL_REGISTRY_API"]+'/api/2.0/mlflow/runs/get?run_id='+os.environ["MODEL_RUN_ID"])
responseData=response.json()

metrics={}
for data in responseData["run"]["data"]["metrics"]:
    metrics[data["key"]] = data["value"]

params={}
for data in responseData["run"]["data"]["params"]:
    params[data["key"]] = data["value"]

tags={}
for data in responseData["run"]["data"]["tags"]:
    tags[data["key"]] = data["value"]
api_data["info"]["description"]= "The API for Disease Progression Modeling Workbench 360: \n \nModel Params: \n \n"+json.dumps(params, separators=(',', ':')) +" \n\nModel  Metrics : \n\n"+json.dumps(metrics, separators=(',', ':'))+" \n\nModel Tags : \n\n"+json.dumps(tags, separators=(',', ':'))

with open(static_api_dir, 'w') as fp:
    json.dump(api_data, fp, indent=2)


"""
Swagger input doc resource
"""
patient_request_fields = api.model('Resource', {
    'patient_id': fields.String
})

"""
Swagger endpoints, which serves static docs from https://github.com/swagger-api/swagger-ui
"""
@app.route("/" + os.environ['MODEL_NAME']+'/api/')
def get_docs():
    print('sending docs')
    return render_template('swaggerui.html')

@app.route("/" + os.environ['MODEL_NAME']+'/static/js/<path>')
def load_static_js_files(path):
    return send_from_directory("./" + os.environ['MODEL_NAME']+'/static/js/',path)

@app.route("/" + os.environ['MODEL_NAME']+'/static/<path>')
def load_static_files(path):
    return send_from_directory("./" + os.environ['MODEL_NAME']+'/static/',path)

"""
Default root
"""
@api.route('/')
@api.doc()
class HelloWorld(Resource):
    def get(self):
        response = api.make_response({'Welcome': 'to the DPM360 api.... :)'}, 200)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Platform, Version'
        response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, TRACE, GET, HEAD, POST, PUT, DELETE'
        return response

"""
Model predict 
"""
@api.route('/model/predict')
class ModelInput(Resource):
    @api.doc(body=patient_request_fields, description="This is the place to use to submit a a request to a given model.  Please provide a json (as the request body) containing patient id as input.",
        responses={
            200: 'Success',
            400: 'Validation Error'
        })
    def post(self):
        """
        Retrieve patient id
        """
        data = request.get_json()
        patientId = data["patient_id"];

        """
        Call model patient predict using with the given patient id
        """
        patient_proba = app.model.predict_patient(patientId, app.model_dataset, app.model)
        response = {"predicted_max_value": T.argmax(patient_proba).tolist(), "patient_id": patientId,"predicted_values": patient_proba.tolist()}
        return response

"""
Model patient id from model dataset 
"""
@api.route('/model/dataset')
@api.doc(description="View model dataset")
class PatientDataset(Resource):
    def get(self):
        """
        Load data and return patient data ids
        """
        return send_from_directory(app.features_local_path, app.model_y_test_file_name)

"""
Start App 
"""
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=os.environ['PORT'])