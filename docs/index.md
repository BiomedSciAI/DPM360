### DPM360 -- Disease Progression Modeling workbench 360, aiming at providing an end-to-end deep learning model training framework in python on OHDSI-OMOP data

#  Overview

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/CLnMzRv0hCc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

Disease Progression Modeling workbench 360 (DPM360) is a clinical informatics framework for collaborative research and delivery of healthcare AI. DPM360, when fully developed, will manage the entire modeling life cycle, from data analysis (e.g., cohort identification) to machine learning algorithm development and prototyping. DPM360 augments the advantages of data model standardization and tooling (OMOP-CDM, Athena, ATLAS) with a powerful machine learning training framework, and a mechanism for rapid prototyping through automatic deployment of models as containerized services to a cloud environment.

<figure><center><img src=./resources/png/dpm360v2.png "DPM360" width="300"/></center><figcaption>DPM360 Component View</figcaption></figure>

## Background

Chronic diseases are becoming more and more prevalent across the globe and are known to drive rising costs of healthcare. The health informatics community has reacted to these challenges with the development of data driven Disease Progression Modeling (DPM) techniques seeking to describe the time course of such diseases to track evolution and predict severity over time. These techniques are vast, ranging from disease staging and patient trajectory analytics, prediction and event time to event estimations for important disease related events of interests. While applications of DPM are numerous for both providers (e.g., decision support for patient care), payers (e.g., care management) and pharmas (e.g., clinical trial enrichment), the adoption of DPM is hindered by the complexity of developing and managing DPM models throughout their life cycle, from data to production. While organizations like [OHDSI](https://www.ohdsi.org/) have made huge strides to help the research community with widely adopted data models like OMOP coupled with cohorting tools like [Atlas](https://www.ohdsi.org/atlas-a-unified-interface-for-the-ohdsi-tools/), work remains to be done to provide the right platform for the complete management of DPM models. In this demonstration, we introduce Disease Progression Modeling Workbench 360 (DPM360), a work-in-progress system to address these concerns.


Building upon the ideas from our [earlier work](https://arxiv.org/abs/2007.12780), DPM360 is compatible with such OHDSI open tools while enabling health informaticians to build and manage DPM models throughout their entire life cycle within modern cloud infrastructures. DPM360 also facilitates the transparent development of such models following best practices embedded in its modeling framework, thus addressing reproducibility challenges that the AI community is facing.

## Design

DPM360 is made up of three key components. 

<figure><center><img src=./resources/png/dpm360-full-arch-current.png width="800"/></center><figcaption>DPM360 Architecture</figcaption></figure>

1. **Lightsaber**: an extensible training framework which provides blueprints for the development of disease progression models. It consists of pipelines for training, hyperparameter fine tuning, model calibration, and evaluation. `lightsaber` comes with a reusable library of state-of-the-art machine and deep learning algorithms for DPM (e.g. LSTM for in-hospital mortality predictions). `lightsaber` is built upon state-of-the-art open source community tools. Without `lightsaber`, for each model and datasets, ML researchers have to write our their own custom training routines (more important for deep learning models where we extend [pytorch-lightning](https://pytorch-lightning.readthedocs.io)), custom data processors to ingest data from extracted cohort, and custom model tracking (inbuilt in `lightsaber` and valid for both sklearn and Pytorch). Also, without `lightsaber` for every model and type of model ML researchers have to use in custom metrics and evaluations - `lightsaber` standardizes and integrates all of these - e.g., for a classification metrics such as `AUC-ROC`, `AUC-PROC`, `Recall@K`, etc are automatically tracked. `lightsaber` also integrates recalling such evaluations for post-hoc report generation and model maintenance by providing routines to interact with Model Registry. Without `lightsaber`, all of these need to be custom built and repeated for each model. It also provides additional built-in utilities for model calibration. In summary, to develop and test a new deep learning model, we need to code:
      1. network architecture and the loss function
      2. trainer to train the model on the training data and use the validation data for optimization of the model
      3. measures to prevent overfitting, such as early stopping
      4. tuning the model to find the optimal hyperparameters
      5. evaluating the model on the test data
      6. saving and deploying the model for later use.
    `lightsaber` isolates all engineering parts of training the model [steps 2-6] from the core model development [step 1] so the researcher needs to only focus on the architecture of the network as well as the loss function. All other steps are provided by the `lightsaber` framework in a standardized way for training/optimizing/evaluating/deploying the model. 

    `lightsaber` integrates naturally with the OHDSI stack. The ATLAS-`lightsaber` integration combines the ease and flexibility of defining standardized cohorts using OHDSI’s ATLAS graphical user interface and the power of `lightsaber`. The integration enables standardized and reproducible implementation of patient-level prediction tasks based on the OMOP CDM and in the Python programming language.
      1. Training data is pulled using the cohort definitions stored in the OMOP data store and OMOP Common Data Model (OMOP CDM) using Python APIs.
      2. The integration will provide a suite of custom queries and feature extraction algorithms for generating and transforming cohort features into formats necessary for complex machine learning tasks such as time-series modeling in ICU settings not currently supported by the OHDSI tools. 
      3. Additionally, to further improve reuse and reproducibility, the integration will provide a standardized naming and versioning convention for features and data extracted from data in the OMOP CDM model and subsequently used in `lightsaber`.
	Tracking provenance of all aspects of model building is essential for reproducibility. Training experiments run using `lightsaber` are automatically tracked in a Model Registry including parameters, metrics and model binaries allowing ML researchers to identify algorithms and parameters that result in the best model performance.

2. The **Service Builder** component automatically converts models registered in the Model Registry into cloud-ready analytic microservices and serves them in the target execution environment (Kubernates or OpenShift) using KFServing. Thereafter users can test and/or interact with the deployed model microservice via a [Swagger](https://swagger.io/) based interface. The service builder will provide intuitive flexibility to make it easy for everyone to develop, train, deploy, serve and scale Machine Learning (ML) models. These capabilities will assist in managing the full lifecycle of ML models by leveraging on various open source projects such as [Kubeflow](https://www.kubeflow.org/). 

3. The **Installer** component installs the fully functional DPM360 into Kubernetes or OpenShift Container Platform using [Helm charts](https://helm.sh/). Upon installation, models for different endpoints are available for the user. Helm Charts are simply Kubernetes manifests combined into a single package that can be installed to Kubernetes clusters. Once packaged, installing a Helm Chart into a cluster is as easy as running a single helm install, which really simplifies the deployment of containerized applications.  

## Code
The code for DPM360 can be found at our <a href="https://github.com/BiomedSciAI/DPM360">Github repository</a>

<!--
## Roadmap
<figure><center><img src=./resources/png/dpm360_roadmap.png /></center><figcaption>DPM360 Roadmap</figcaption></figure>
-->

## Contribute
We love to hear from you by asking question, reporting bugs, feature requests and contributing as a committer to shape the future of the project. 

We use [GitHub issues](https://github.com/BiomedSciAI/DPM360/issues) for tracking requests and bugs.

## Team
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=ke-NELSONBO">Nelson Bore</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-ssaranathan">Sundar Saranathan</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-ibuleje">Italo Buleje</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=ibm-Prithwish.Chakraborty">Prithwish Chaktraborty</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-rachitac">Rachita Chandra</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-deysa">Sanjoy Dey</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-ekeyigoz">Elif K Eyigoz</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=ibm-Mohamed.Ghalwash">Mohamed Ghalwash</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=jp-AKOSEKI">Akira Koseki</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=ibm-Piyush.Madan1">Piyush Madan</a> 
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-mahatma">Shilpa Mahatma</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=ibm-William.Ogallo">William Ogallo</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=ke-Sekou.Lionel.Remy">Sekou Remy</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-pmeyerr">Pablo Meyer Rojas</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-sowdaby">Daby Sow</a>
* <a href="https://researcher.watson.ibm.com/researcher/view.php?person=us-psuryan">Parthasarathy Suryanarayanan</a>

## Publications
If you use DPM360, please cite us as below:
<br />
Suryanarayanan, P., Chakraborty, P., Madan, P., Bore, K., Ogallo, W., Chandra, R., Ghalwash, M., Buleje, I., Remy, S., Mahatma, S. and Meyer, P., 2021. Disease Progression Modeling Workbench 360. arXiv preprint <a href="https://arxiv.org/abs/2106.13265">arXiv:2106.13265.</a>

##Related Publications


<a href="https://www.medrxiv.org/content/10.1101/2021.03.15.21253549v1">
Dey, S., Bose, A., Chakraborty, P., Ghalwash, M., Saenz, A. G., Ultro, F., Kenney, N., Hu, J., Parida, L., and Sow, D.. Impact of Clinical and Genomic Factors on SARS-CoV2 Disease Severity. 
Accepted at AMIA 2021 Annual Symposium, To appear. (2021)
</a>

## License
[Apache License 2.0](https://github.com/BiomedSciAI/DPM360/LICENSE.txt)

