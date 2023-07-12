# Graph Neural Networks for the prediction of infinite dilution activity coefficients

[![DOI](https://zenodo.org/badge/406258262.svg)](https://zenodo.org/badge/latestdoi/406258262)

<img src="https://github.com/edgarsmdn/GNN_IAC/blob/main/GNN_IAC_logo.png" width="300">

## Description

This repository contains the training routines and the experiments presented in the paper [Graph Neural Networks for the prediction of infinite dilution activity coefficients](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d1dd00037c#!divAbstract).

To cite this work:

```
@Article{D1DD00037C,
author ="Sanchez Medina, Edgar Ivan and Linke, Steffen and Stoll, Martin and Sundmacher, Kai",
title  ="Graph neural networks for the prediction of infinite dilution activity coefficients",
journal  ="Digital Discovery",
year  ="2022",
volume  ="1",
issue  ="3",
pages  ="216-225",
publisher  ="RSC",
doi  ="10.1039/D1DD00037C",
url  ="http://dx.doi.org/10.1039/D1DD00037C"}

```

### GNN_IAC

The folder `GNN_whole_dataset` containes the training routines and trained GNNs on the complete dataset consisting of 2810 different binary systems. Reports for the prediction performance are also included along with the predictions themselves. 

### Hybrid models

The GNNs trained on the corresponding feasible data are contained in each of the `0#_Name_of_model` folders. There, you can also find the GNNs trained on the corresponding residuals. The training routines are also included. For each case, a report with the prediction statistics is also given.

## Important Notice! ⚠️

Our original work makes use of a dataset originally retrieved and published by [Brouwer et al. (2019)](https://doi.org/10.1021/acs.iecr.9b00727). On April 2023, [a correction](https://doi.org/10.1021/acs.iecr.3c00757) was published on this same dataset noticing that out of the 5194 data points in the original dataset 24 entries contained errors. In 18 of them
iodomethane was incorrectly used instead of diiodomethane. The other 6 contained a typo of incorrect placement of the decimal point in the experimental value of the infinite dilution activity coefficient.

We have re-run our computational experiments and analysis to observe the impact of this correction in our original work. Overall, the correction have led to minor implications on the models accuracy and coverage percentage. The accuracy of the GNN-based models was in general slightly improved. Moreover, the original analysis, general conclusions and findings remain the same.

You can see the impact report [here](https://github.com/edgarsmdn/GNN_IAC/blob/main/Correction_notice.pdf).

## Requirements

The following libraries need to be also installed:
* [PyTorch](https://pytorch.org/) >= 1.8.0
* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) >= 2.0
* [RDKiT](https://www.rdkit.org/docs/index.html) >= 2021.03.1

## License

This material is licensed under the [MIT license](https://github.com/edgarsmdn/GNN_IAC/blob/main/LICENSE) and is free and provided as-is.

## Link
https://github.com/edgarsmdn/GNN_IAC
