
# SCAPEXNet Experiments

This repository contains the experimental code associated with the manuscript:

**A Structured and Scale-Conditioned Neural Parameterization for Subgrid Tracer Fluxes in Coarse-Resolution Ocean Models**

SCAPEXNet is a physics-guided neural parameterization for predicting unresolved subgrid-scale temperature and salinity fluxes in coarse-resolution ocean models.

## Data Source

The dataset is derived from eddy-resolving simulations produced by the **Institute of Atmospheric Physics Climate System Ocean Model version 3 (LICOM3)**.

The original daily outputs include:

- potential temperature (`T`)
- salinity (`S`)
- zonal velocity (`u`)
- meridional velocity (`v`)
- latitude and longitude coordinates

The original data are stored as aligned NetCDF fields on the native curvilinear ocean grid. Area-weighted horizontal coarse-graining is applied at factors `n = 3`, `5`, and `10`.

The prediction targets are the four subgrid tracer-flux components:

- zonal temperature flux
- meridional temperature flux
- zonal salinity flux
- meridional salinity flux

The full LICOM3 dataset is not included in this repository. Before running the scripts, users should configure the local data paths according to their LICOM3 data location.

## Environment

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate the environment using the environment name defined in the first line of `environment.yml`:

```bash
conda activate <environment-name>
```

## Repository Structure and Experiment Mapping

| File | Corresponding experiment |
|---|---|
| `analysisFile/analy_1.py` | General result analysis and summary |
| `exp1/exp1_art_1.py` | Experiment 1: multi-scale benchmarking across coarse-graining factors |
| `exp2/exp2_art_1.py` | Experiment 2: mechanism diagnostics of SCAPEXNet across scales |
| `exp3/exp3_art_1.py` | Experiment 3: ablation study of SCAPEXNet |
| `exp3/exp3_art_plot.py` | Plotting and visualization for Experiment 3 |
| `exp4/exp4_art_1.py` | Experiment 4: physical robustness and structural consistency under expert simplification |
| `exp5/exp5_art_1.py` | Experiment 5: day-level robustness assessment |
| `exp6/exp6_art_1.py` | Experiment 6: reduced-budget sensitivity analysis of control parameters |
| `exp6/exp6_art_3.py` | Additional parameter-sensitivity analysis for Experiment 6 |
| `exp_1.py` | Shared or supplementary experiment script |

## Running the Experiments

After configuring the data paths, each experiment can be run from the repository root. For example:

```bash
python exp1/exp1_art_1.py
```

Other experiments can be executed in the same way using their corresponding scripts.

## Notes

The experiments use the same general data-processing and evaluation framework, including day-level data splitting, ocean masking, multi-scale coarse-graining, and evaluation of the four temperature and salinity flux components.

For exact training parameters and output locations, refer to the configuration section inside each experiment script.
