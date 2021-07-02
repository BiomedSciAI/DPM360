import logging
import os
from pathlib import Path
import mlflow
import pandas as pd
import torch as T
from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
from flask_restplus import Api, Resource, fields
from sklearn.preprocessing import StandardScaler
from lightsaber.data_utils import pt_dataset as ptd
import lightsaber.data_utils.utils as du
from mlflow.tracking import MlflowClient

log = logging.getLogger()
description_text = 'The API for Disease Progression Modeling Workbench 360'

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



class ModeWrapperApp(Flask):
  def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):

    #   Get env variables
    self.modelName = os.environ['MODEL_NAME']
    self.modelVersion = os.environ['MODEL_VERSION']
    self.runId = os.environ["MODEL_RUN_ID"]
    self.modelSource = os.environ["MODEL_SOURCE"]

    # Fetch model from mlflow registry
    # NB: We are fetching from local because the wrapped model is too big to do mlflow.pytorch.load_model
    if ("Test" in self.modelName):
        self.model = mlflow.sklearn.load_model(model_uri=self.modelSource)
    else:
        self.model = mlflow.pytorch.load_model(model_uri=self.modelSource)

        # Download artifacts from mlflow
        self.mlflowClient =MlflowClient()
        local_dir = os.getcwd()
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        local_path = self.mlflowClient.download_artifacts(self.runId, "features", local_dir)

        print("Artifacts downloaded in: {}".format(local_path))
        print("Artifacts: {}".format(os.listdir(local_path)))

        """
        The following section is what is needed by lightsaber code, we will move this to the model at a later point
        """
        data_dir = (Path.cwd() / './features').resolve()
        expt_conf = du.yaml.load(
            open(Path.cwd() / './features/ohdsi_ihm_expt_config.yml').read().format(DATA_DIR=data_dir),
            Loader=du._Loader)

        preprocessor = StandardScaler()
        train_filter = [filter_preprocessor(cols=expt_conf['numerical'],
                                                            preprocessor=preprocessor,
                                                            refit=True),
                        filter_fillna(fill_value=expt_conf['normal_values'],
                                                      time_order_col=expt_conf['time_order_col'])
                        ]
        transform = ptd.transform_drop_cols(cols_to_drop=expt_conf['time_order_col'])

        self.test_dataset = ptd.BaseDataset(tgt_file=expt_conf['test']['tgt_file'],
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

  # Given the patient id, find the array index of the patient
  def predict_patient(self, patientid,test_dataset, model):

      self.p_idx = test_dataset.sample_idx.index.get_loc(patientid)
      self.p_x, self.p_y, self.p_lengths, _ = test_dataset[self.p_idx]
      self.p_x.unsqueeze_(0)  # adding dummy dimension for batch
      self.p_y.unsqueeze_(0)  # adding dimension for batch
      p_lengths = [self.p_lengths, ]

      proba = model.predict_proba(self.p_x, lengths=p_lengths)
      return proba


app = ModeWrapperApp(__name__)
api = Api(app, version='1.0', title='DPM360 API', prefix="/api/v1",description=description_text, doc='/doc/')
auth = HTTPBasicAuth()
workflow_request_fields = api.model('Resource', {
    'patient_id': fields.String
})
workflow_status_fields = api.model('ModelInput', {
    'modelRunID': fields.String
})

@api.route('/')
@api.doc()
class HelloWorld(Resource):
    def get(self):
        response = api.make_response({'Welcome': 'to the DPM360 api.... :)'}, 200)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Platform, Version'
        response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, TRACE, GET, HEAD, POST, PUT, DELETE'
        return response

@api.route('/model/predict')
class ModelInput(Resource):
    @api.doc(body=workflow_request_fields, description="This is the place to use to submit a a request to a given model.  Please provide a json (as the request body) containing model input.",
        responses={
            200: 'Success',
            400: 'Validation Error'
        })
    def post(self):
        data = request.get_json()
       # We check if its test model if not use lightsaber model
        if("Test" in app.modelName):
            transformedData = pd.DataFrame(data)
            results = app.model.predict(transformedData)
            response = {"predicted_value": results[0]}
            return response
        else:
            # get patient id
            patientId = data["patient_id"];

            # patient predict
            proba = app.predict_patient(patientId, app.test_dataset, app.model)
            response = {"predicted_value": str(T.argmax(proba)),"actual_value": str(app.p_y.squeeze().data.cpu().numpy())}
            return response

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9090)
