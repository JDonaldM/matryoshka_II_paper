# [matryoshka](https://github.com/JDonaldM/Matryoshka) II: Accelerating Effective Field Theory Analyses of the Galaxy Power Spectrum

This is the accompanying repository to the paper [Donald-McCann, Koyama, Beutler (2022)](https://arxiv.org/abs/2202.07557). It contains all mock data, all MCMC chains, and all importance weights that form the results of the paper. It also includes all the `Python` code needed to conduct the analyses presented in the paper. Most of this code is presented in the form of Jupyter notebooks.

## Requirements

The following packages (and their corresponding dependencies) are required to run the code in this repo. The version numbers in parentheses are those that were used to produce the results presented in the paper and contained within this repo. Using different versions may lead to different results.

- [matryoshka](https://github.com/JDonaldM/Matryoshka) (0.2.7)
- [PyBird](https://github.com/pierrexyz/pybird) (0.1)
- [zeus](https://github.com/minaskar/zeus) (2.4.1)
- [corner](https://github.com/dfm/corner.py) (2.2.1)
- [GetDist](https://github.com/cmbant/getdist) (1.2.1)

## Changes in v2.0.0
During the review process of the paper there were significant changes made to this repo. Almost every file has been change, although the conclusions drawn in the paper remain the same. The most significant change involved calculating importance weights for more than one of the mock analysis setups. As such the code related to this has been change from a notebook to a `Python` script.

## Running order

**Setup**

1. [./notebooks/gen_training_cosmo.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/gen_training_cosmo.ipynb): This notebook will generate some samples from the five parameter LCDM training space. 
2. [Matryoshka/scripts/genEFTEMUtraindata.py](https://github.com/JDonaldM/Matryoshka/blob/main/scripts/genEFTEMUtraindata.py): This will calculate the P<sub>n</sub> components that will be emulated. To run the full set of analyses this will need to be run for z = [0.38, 0.51, 0.61]. **This will take some time!** It is by far the most expensive step, it took ~8hr per redshift on an old i5.
3. [Matryoshka/scripts/trainEFTEMUcomponents.py](https://github.com/JDonaldM/Matryoshka/blob/main/scripts/trainEFTEMUcomponents.py): This will train all the component emulators of the `EFTEMU`. This will again need to be done for each redshift. Should take ~30min per redshift.
4. [./notebooks/make_mocks_lin_and_loops.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/make_mocks_lin_and_loops.ipynb): This will calculate the P<sub>n</sub> components for the mock data.
5. [./notebooks/make_mocks_cov_and_poles.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/make_mocks_cov_and_poles.ipynb): This will calculate the mock multipoles and Gaussian covariances.

**Analysis**

6. [./scripts/runMCMCwEFTEMU.py](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/scripts/runMCMCwEFTEMU.py): This will run an MCMC with the `EFTEMU`. This needs to be done for every mock setup. We provide a [notebook](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/mcmc_z0.61_v3700_EFTEMU.ipynb) that explains some of the code in the script.
7. [./scripts/pybird_importance_likes_v2.py](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/scripts/pybird_importance_likes_v2.py): This `Python` script will calculate `PyBird` likelihoods for a 'chunk' of samples from a chain run with the `EFTEMU`. Each chunk is 10000 samples. Likelihoods for at least one chunk need to be calculated  
8. [./scripts/compute_weights.py](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/scripts/compute_weights.py): This script will compute the importance weights. Needs to be done for each set of `PyBird` likelihoods.
9. [./notebooks/performance.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/performance.ipynb): This notebook compares the compuational performance of the `EFTEMU` to `PyBird`.

**Plotting**

10. [./notebooks/per_err.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/per_err.ipynb): This notebook plots the power spectrum level prediction accuracy. Figure 3.
11. [./notebooks/isnr_P0.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/isnr_P0.ipynb): This notebook calculates the inverse signal-to-noise ratio for the monopole for each of the mock volumes. Figure 4.
12. [./notebooks/map_plot_V5000.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/map_plot_V5000.ipynb): This notebook plots the MAP predictions. Figure 5.
13. [./notebooks/nice_corners_samples.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/nice_corners_samples.ipynb): This notebook produces the corner plot for the posteriors at each redshift. Figure 6.
14. [./notebooks/nice_corners--z-0.38.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/nice_corners--z-0.38.ipynb): This notebook compares the `PyBird` and `EFTEMU` posteriors. Figures 7 and 8.

## Attribution

Please cite [Donald-McCann, Koyama, Beutler (2022)](https://arxiv.org/abs/2202.07557) if you found this code useful in your research:

```bash
@article{donald-mccann2022matryoshkaII,
  title={matryoshka II: Accelerating Effective Field Theory Analyses of the Galaxy Power Spectrum},
  author={Donald-McCann, Jamie and Koyama, Kazuya and Beutler, Florian},
  journal={arXiv preprint arXiv:2202.07557},
  year={2022}
}
```
