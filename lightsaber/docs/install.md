# Standalone Installation

`Lightsaber` is an integral part of the overall DPM360 pipeline. However, by design, `Lightsaber` can also be used standalone as a python package. 

## Installation Instructions

<<<<<<< HEAD
`Lightsaber` is installable as a python package. 

It can be installed using `conda` as:
```
conda install -c conda-forge dpm360-lightsaber
```
 or from `pypi` as:
```
pip install dpm360-lightsaber
```
It can also be installed from source using pip as follows:
=======
`Lightsaber` is installable as a python package from source using `pip`. (currently,  its not on `pypi`/`conda`) as follows:

>>>>>>> fb1fecce5da53ce2c45bfb57f8fcd95e4c43f885
* barebones install of `Lightsaber`: `pip install .` 
* with doc support: `pip install .[doc]`
* with time-to-event modeling (T2E) support: `pip install .[t2e]`
* full install with all components: `pip install .[full]`

For convenience, an example conda environment compatible with `Lightsaber` is available in [github](https://github.com/IBM/DPM360/blob/main/environment.yaml).
