# GCS-CAM

[![][license-img]][license-status] [![][zenodo-img]][zenodo-status]

Code to partially reproduce results in "[De-risking geological carbon storage from high resolution time-lapse seismic to explainable leakage detection](https://slim.gatech.edu/content/de-risking-geological-carbon-storage-high-resolution-time-lapse-seismic-explainable-leakage)"

## Installation

Run the command below to install the required packages.

```bash
julia -e 'Pkg.add("DrWatson.jl")'
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Script descriptions

`GenLinData.jl`: script to generate time-lapse linearized data via Born modeling operators.

`RTM.jl`: script to run reverse-time migration (RTM) on the linearized data.

`JRM.jl`: script to invert the time-lapse linearized data via joint recovery model (JRM).

The experimental setup (number of sources, receivers, amount of noise etc) can be adjusted according to [input keywords](https://github.com/slimgroup/GCS-CAM/blob/main/src/utils.jl).

To generate a dataset for training the deep neural classifier, we provide the clusterless version of the above 3 scripts --- where you can simply run the julia scripts locally and experiments can run on multiple instances in parallel on the cloud. This needs 3 files for registry, credential, and parameter information to be stored in `registryinfo.json`, `credentials.json`, `params.json` files. More information can be found in [AzureClusterlessHPC.jl](https://github.com/microsoft/AzureClusterlessHPC.jl) and [JUDI4Cloud.jl](https://github.com/slimgroup/JUDI4Cloud.jl).

## LICENSE

The software used in this repository can be modified and redistributed according to [MIT license](https://github.com/slimgroup/GCS-CAM/blob/main/LICENSE).

## Reference

If you use our software for your research, please cite our preprint:

```bibtex
@article {yin2022TLEdgc,
	title = {De-risking geological carbon storage from high resolution time-lapse seismic to explainable leakage detection},
	journal = {The Leading Edge},
	year = {2022},
	note = {Just accepted in the January 2023 special section in seismic resolution},
	month = {09},
	url = {https://slim.gatech.edu/Publications/Public/Journals/TheLeadingEdge/2022/yin2022TLEdgc/paper.html},
	software = {https://github.com/slimgroup/GCS-CAM},
	author = {Ziyi Yin and Huseyin Tuna Erdinc and Abhinav Prakash Gahlot and Mathias Louboutin and Felix J. Herrmann}
}
```

## Authors

This package was written by Ziyi Yin, Huseyin Tuna Erdinc, Abhinav Prakash Gahlot from the [Seismic Laboratory for Imaging and Modeling](https://slim.gatech.edu/) (SLIM) at the Georgia Institute of Technology.

If you have any question, we welcome your contributions to our software by opening issue or pull request.

SLIM Group @ Georgia Institute of Technology, [https://slim.gatech.edu](https://slim.gatech.edu/).      
SLIM public GitHub account, [https://github.com/slimgroup](https://github.com/slimgroup).    

[license-status]:LICENSE
[zenodo-status]:https://doi.org/10.5281/zenodo.7222318
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
[zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.3878711.svg?style=plastic