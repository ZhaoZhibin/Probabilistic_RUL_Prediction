
## Probabilistic_RUL_Prediction

Code release for **[Probabilistic Remaining Useful Life Prediction Based on Deep Convolutional Neural Network](http://dx.doi.org/10.2139/ssrn.3717738)** by [Zhibin Zhao](https://zhaozhibin.github.io/).

## Abstract
Remaining useful life (RUL) prediction plays a vital role in prognostics and health management (PHM) for improving the reliability and reducing the cycle cost of numerous mechanical systems. Deep learning (DL) models, especially deep convolutional neural networks (DCNNs), are becoming increasingly popular for RUL prediction, whereby state-of-the-art results have been achieved in recent studies. Most DL models only provide a point estimation of the target RUL, but it is highly desirable to have associated confidence intervals for any RUL estimate. To improve on existing methods, we construct a probabilistic RUL prediction framework to estimate the probability density of target outputs based on parametric and non-parametric approaches. The model output is an estimate of the probability density of the target RUL, rather than just a single point estimation. The main advantage of the proposed method is that the method can naturally provide a confidence interval (aleatoric uncertainty) of the target prediction. We verify the effectiveness of our constructed framework via a simple DCNN model on a publicly available degradation simulation dataset of turbine engines.

## Guide
This codes contain three different training modes, including MSE (using the MSE loss to train the model), QL (using the non-parametric approach to train the model) and GD (using the parametric approach to train the model)

Meanwhile, all the experiments are executed under Window 10 and Pytorch 1.3 through running on a computer with an Intel Core i7-9700K, GeForce RTX 2080Ti, and 16G RAM.


## Requirements
- Python 3.7
- Numpy 1.16.2
- Pandas 0.24.2
- Pickle
- tqdm 4.31.1
- sklearn 0.21.3
- Scipy 1.2.1
- pytorch >= 1.1
- torchvision >= 0.40


## Datasets
- **[Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)**


## Pakages

This repository is organized as:
- [loss](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/loss) contains different loss functions.
- [datasets](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/datasets) contains the normalization methods and the Pytorch datasets.
- [models](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/models) contains the models used in this project.
- [utils](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/utils) contains the functions for realization of the training procedure.


## Usage
- download datasets

- use the Generate_C-MAPSS.py to prepare and generate the data for the training and testing phases.

- use the train_dlc_resnet_single_time_QL.py to training the model.

- use the test_single_model.py to test the samples for saved model.

- use the plot_log.py to plot the predicted RUL and its corresponding confidence interval.


## Citation
Codes:
```
@misc{Zhao2019,
author = {Zhibin Zhao and Jingyao Wu and David Wong and Chuang Sun and Ruqiang Yan},
title = {Probabilistic Remaining Useful Life Prediction Based on Deep Convolutional Neural Network},
year = {2020},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/ZhaoZhibin/Probabilistic_RUL_Prediction}},
}
```
Paper:
```
@article{zhao2019unsupervised,
  title={Probabilistic Remaining Useful Life Prediction Based on Deep Convolutional Neural Network},
  author={Zhibin Zhao and Jingyao Wu and David Wong and Chuang Sun and Ruqiang Yan},
  journal={TESConf 2020 - 9th International Conference on Through-life Engineering Services},
  year={2020}
}
```
## Contact
- zhibinzhao1993@gmail.com
