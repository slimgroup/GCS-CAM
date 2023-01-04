<h1 align="center">Derisking geological carbon storage from high-resolution time-lapse seismic to explainable leakage detection</h1>

[![][license-img]][license-status] [![][zenodo-img]][zenodo-status]

Code to reproduce results in Ziyi Yin, Huseyin Tuna Erdinc, Abhinav Prakash Gahlot, Mathias Louboutin, Felix J. Herrmann, "[Derisking geological carbon storage from high-resolution time-lapse seismic to explainable leakage detection](https://arxiv.org/abs/2211.03527)". In The Leading Edge in January 2023. DOI: [10.1190/tle42010069.1](https://doi.org/10.1190/tle42010069.1)

## Installation

First, install [Julia](https://julialang.org/), [Python](https://www.python.org/) and [MiniConda](https://docs.conda.io/en/latest/miniconda.html). Next, run the command below to install the required packages.

```bash
julia -e 'Pkg.add("DrWatson.jl")'
julia --project -e 'using Pkg; Pkg.instantiate()'
conda env create -f environment.yml
source activate gcs-cam
python -m ipykernel install --user --name gcs-cam --display-name "Python (gcs-cam)"
```

## Script descriptions

We use the open-source software [JUDI.jl](https://github.com/slimgroup/JUDI.jl) for seismic modeling and imaging, which calls the highly optimized propagators of [Devito](https://www.devitoproject.org/). We used [FwiFlow.jl](https://github.com/lidongzh/FwiFlow.jl) to solve the two-phase flow equations for both the pressure and concentration. The CO2 plume dataset (consisting of regular plumes and leaking plumes) will be downloaded upon running your first example. We used [PyTorch library for CAM methods](https://github.com/jacobgil/pytorch-grad-cam) to calculate the CAM images. We thank the authors of these packages for their contributions to the open-source software community.

### time-lapse seismic modeling and imaging

`GenLinData.jl`: script to generate time-lapse linearized data via Born modeling operators.

`RTM.jl`: script to run reverse-time migration (RTM) on the linearized data.

`JRM.jl`: script to invert the time-lapse linearized data via joint recovery model (JRM).

The experimental setup (number of sources, receivers, amount of noise etc) can be adjusted according to [input keywords](src/utils.jl).

To generate a dataset for training the deep neural classifier, we provide the clusterless version of the above 3 scripts --- where you can simply run the julia scripts locally and experiments can run on multiple instances in parallel on the cloud. This needs 3 files for registry, credential, and parameter information to be stored in `registryinfo.json`, `credentials.json`, `params.json` files. More information can be found in [AzureClusterlessHPC.jl](https://github.com/microsoft/AzureClusterlessHPC.jl) and [JUDI4Cloud.jl](https://github.com/slimgroup/JUDI4Cloud.jl).

### leakage detection with deep neural classifier and class activation mapping

To train the deep neural classifier for leakage detection, open `main.ipynb` notebook and choose `gcs-cam` environment as the kernel. It internally uses `train.py` and `test.py` modules for training and testing. The notebook contains useful comments for each section.

## LICENSE

The software used in this repository can be modified and redistributed according to [MIT license](LICENSE).

## Reference

If you use our software for your research, we appreciate it if you cite us following the bibtex in [CITATION.bib](CITATION.bib).

## Authors

This repository is written by Ziyi Yin, Huseyin Tuna Erdinc, Abhinav Prakash Gahlot from the [Seismic Laboratory for Imaging and Modeling](https://slim.gatech.edu/) (SLIM) at the Georgia Institute of Technology.

If you have any question, we welcome your contributions to our software by opening issue or pull request.

SLIM Group @ Georgia Institute of Technology, [https://slim.gatech.edu](https://slim.gatech.edu/).      
SLIM public GitHub account, [https://github.com/slimgroup](https://github.com/slimgroup).    

[license-status]:LICENSE
[zenodo-status]:https://doi.org/10.5281/zenodo.7222318
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
[zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.3878711.svg?style=plastic
