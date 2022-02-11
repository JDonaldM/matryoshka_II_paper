# [matryoshka](https://github.com/JDonaldM/Matryoshka) II: Accelerating Effective Field Theory Analyses of the Galaxy Power Spectrum

This is the accompanying repository to the paper Donald-McCann, Koyama, Beutler (2022). It contains all mock data, all MCMC chains, and all importance weights that form the results of the paper. It also includes all the `Python` code needed to conduct the analyses presented in the paper. Most of this code is presented in the form of Jupyter notebooks.

## Requirements

The following packages (and their corresponding dependencies) are required to run the code in this repo. The version numbers in parentheses are those that were used to produce the results presented in the paper and contained within this repo. Using different versions may lead to different results.

- [matryoshka](https://github.com/JDonaldM/Matryoshka) (0.2.7)
- [PyBird](https://github.com/pierrexyz/pybird) (0.1)
- [zeus](https://github.com/minaskar/zeus) (2.4.1)
- [corner](https://github.com/dfm/corner.py) (2.2.1)
- [GetDist](https://github.com/cmbant/getdist) (1.2.1)

## Running order

**Setup**

1. [./notebooks/gen_training_cosmo.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/gen_training_cosmo.ipynb): This notebook will generate some samples from the five parameter LCDM training space. 
2. [Matryoshka/scripts/genEFTEMUtraindata.py](https://github.com/JDonaldM/Matryoshka/blob/main/scripts/genEFTEMUtraindata.py): This will calculate the P<sub>n</sub> components that will be emulated. To run the full set of analyses this will need to be run for z = [0.38, 0.51, 0.61]. **This will take some time!** It is by far the most expensive step, it took ~8hr per redshift on an old i5.
3. [Matryoshka/scripts/trainEFTEMUcomponents.py](https://github.com/JDonaldM/Matryoshka/blob/main/scripts/trainEFTEMUcomponents.py): This will train all the component emulators of the `EFTEMU`. This will again need to be done for each redshift. Should take ~30min per redshift.

**Analysis**

4. [./notebooks/make_mocks_lin_and_loops.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/make_mocks_lin_and_loops.ipynb): This will calculate the P<sub>n</sub> components for the mock data.
5. [./notebooks/make_mocks_cov_and_poles.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/make_mocks_cov_and_poles.ipynb): This will calculate the mock multipoles and Gaussian covariances.
6. [runMCMCwEFTEMU.py](): This will run an MCMC with the `EFTEMU`. This needs to be done for each mock with V<sub>s</sub> = 3700 Mpc *h*<sup>-1</sup>, and for the mock at z = 0.61 for every mock volume. Should take ~30mins per MCMC. The importance sampling also requires multiple MCMCs at z = 0.61 with V<sub>s</sub> = 3700 Mpc *h*<sup>-1</sup> to boost the effective sample size. We provide a [notebook](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/mcmc_z0.61_v3700_EFTEMU.ipynb) that explains some of the code in the script.
7. [./notebooks/pybird_importance_likes.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/pybird_importance_likes.ipynb): This notebook will compute the `PyBird` likelihood for samples in a thinned `EFTEMU` chain. Should take ~1.5hr per chain.
8. [./notebooks/importance_z0.61_Vc3700_Vi5000.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/importance_z0.61_Vc3700_Vi5000.ipynb): This notebook will compute the importance weights.

**Plotting**

9. [./notebooks/per_err.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/per_err.ipynb): This notebook plots the power spectrum level prediction accuracy. Figure 2.
10. [./notebooks/map_plot_V3700.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/map_plot_V3700.ipynb): This notebook plots the MAP predictions. Figure 3.
11. [./notebooks/nice_corners_samples.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/nice_corners_samples.ipynb): This notebook produces the corner plot for the posteriors at each redshift. Figure 4.
12. [./notebooks/violins--volumes.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/violins--volumes.ipynb): This notebook produces the violin plot for the marginalised 1D posteriors with each V<sub>s</sub>. Figure 5.
13. [./notebooks/nice_corners_pybird.ipynb](https://github.com/JDonaldM/matryoshka_II_paper/blob/main/notebooks/nice_corners_pybird.ipynb): This notebook compares the `PyBird` and `EFTEMU` posteriors. Figure 6.

## Attribution

Please cite Donald-McCann, Koyama, Beutler (2022) if you found this code useful in your research:

```bash
@article{donald-mccann2022matryoshkaII,
  title={matryoshka II: Accelerating Effective Field Theory Analyses of the Galaxy Power Spectrum},
  author={Donald-McCann, Jamie and Koyama, Kazuya and Beutler, Florian},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2022}
}
```
