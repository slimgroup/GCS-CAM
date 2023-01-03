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

The dataset will be downloaded upon running your first example.

### time-lapse seismic modeling and imaging

`GenLinData.jl`: script to generate time-lapse linearized data via Born modeling operators.

`RTM.jl`: script to run reverse-time migration (RTM) on the linearized data.

`JRM.jl`: script to invert the time-lapse linearized data via joint recovery model (JRM).

The experimental setup (number of sources, receivers, amount of noise etc) can be adjusted according to [input keywords](src/utils.jl).

To generate a dataset for training the deep neural classifier, we provide the clusterless version of the above 3 scripts --- where you can simply run the julia scripts locally and experiments can run on multiple instances in parallel on the cloud. This needs 3 files for registry, credential, and parameter information to be stored in `registryinfo.json`, `credentials.json`, `params.json` files. More information can be found in [AzureClusterlessHPC.jl](https://github.com/microsoft/AzureClusterlessHPC.jl) and [JUDI4Cloud.jl](https://github.com/slimgroup/JUDI4Cloud.jl).

### leakage detection with deep neural classifier and class activation mapping

To train the deep neural classifier for leakage detection, open `main.ipynb` notebook and choose `gcs-cam` environment as the kernel. It internally uses `train.py` and `test.py` modules for training and testing. The notebook contains useful comments for each section.

## LICENSE

The software used in this repository can be modified and redistributed according to [MIT license](https://github.com/slimgroup/GCS-CAM/blob/main/LICENSE).

## Reference

If you use our software for your research, please cite our preprint:

```bibtex
@ARTICLE{yin2022TLEdgc,
  author = {Ziyi Yin and Huseyin Tuna Erdinc and Abhinav Prakash Gahlot and Mathias Louboutin and Felix J. Herrmann},
  title = {De-risking geological carbon storage from high resolution time-lapse seismic to explainable leakage detection},
  journal = {The Leading Edge},
  year = {2023},
  month = {01},
  volume = {42},
  number = {1},
  pages = {69–76},
  abstract = {Geological carbon storage represents one of the few truly scalable technologies capable of reducing the {CO$_2$} concentration in the atmosphere. While this technology has the potential to scale, its success hinges on our ability to mitigate its risks. An important aspect of risk mitigation concerns assurances that the injected {CO$_2$} remains within the storage complex. Amongst the different monitoring modalities, seismic imaging stands out with its ability to attain high resolution and high fidelity images. However, these superior features come, unfortunately, at prohibitive costs and time-intensive efforts potentially rendering extensive seismic monitoring undesirable. To overcome this shortcoming, we present a methodology where time-lapse images are created by inverting non-replicated time-lapse monitoring data jointly. By no longer insisting on replication of the surveys to obtain high fidelity time-lapse images and differences, extreme costs and time-consuming labor are averted. To demonstrate our approach, hundreds of noisy time-lapse seismic datasets are simulated that contain imprints of regular {CO$_2$} plumes and irregular plumes that leak. These time-lapse datasets are subsequently inverted to produce time-lapse difference images used to train a deep neural classifier. The testing results show that the classifier is capable of detecting {CO$_2$} leakage automatically on unseen data and with a reasonable accuracy.},
  keywords = {Seismic Imaging, JRM, CCS, classification, CAM, explainability, time-lapse, resolution},
  doi = {10.1190/tle42010069.1},
  note = {(The Leading Edge)},
  software = {https://github.com/slimgroup/GCS-CAM},
  url = {https://slim.gatech.edu/Publications/Public/Journals/TheLeadingEdge/2022/yin2022TLEdgc/paper.html}
}
```

## Authors

This repository is written by Ziyi Yin, Huseyin Tuna Erdinc, Abhinav Prakash Gahlot from the [Seismic Laboratory for Imaging and Modeling](https://slim.gatech.edu/) (SLIM) at the Georgia Institute of Technology.

If you have any question, we welcome your contributions to our software by opening issue or pull request.

SLIM Group @ Georgia Institute of Technology, [https://slim.gatech.edu](https://slim.gatech.edu/).      
SLIM public GitHub account, [https://github.com/slimgroup](https://github.com/slimgroup).    

[license-status]:LICENSE
[zenodo-status]:https://doi.org/10.5281/zenodo.7222318
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
[zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.3878711.svg?style=plastic
