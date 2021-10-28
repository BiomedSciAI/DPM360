# Standalone Installation

`Lightsaber` is an integral part of the overall DPM360 pipeline. However, by design, `Lightsaber` can also be used standalone as a python package. 

## Installation Instructions

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
* barebones install of `Lightsaber`: `pip install .` 
* with doc support: `pip install .[doc]`
* with time-to-event modeling (T2E) support: `pip install .[t2e]`
* full install with all components: `pip install .[full]`

For convenience, an example conda environment compatible with `Lightsaber` is available in [github](https://github.com/IBM/DPM360/blob/main/environment.yaml).
