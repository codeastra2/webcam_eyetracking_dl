# Webcam Eye tracking with Deep Learning

## Introduction
This project provides deep learning models including the pytorch implementation of [Geddnet](https://github.com/czk32611/GEDDnet), for webcam eye tracking. 


## Dataset Prepearation 
Coming Soon...

## Model Training
If you wish change the hyperparameters such as epochs or learning rate, it can be done so in the file 
[hyperparams.json](https://github.com/codeastra2/Webcam_Eyetracking/blob/main/model_training/webcam_eyetracking/model_training_config/hyperparams.json).
Run the following commands to train the model, with a GPU it should take ~45 mins for 30 epochs of training. 


- `cd model_training`
- `python -m  webcam_eyetracking.train_model`

## Inference

The results of the train/test can be found in the `reports` folder. Here you can find various metrics such as loss curves, metric curves heatmaps representing the genral prediction and error distribution. 

TBD: Add some images here. 

## Live Demo

Coming soon.... 

## References

- Chen, Zhaokang, and Bertram Shi. “Towards High Performance Low Complexity Calibration in Appearance Based Gaze Estimation.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022, pp. 1–1. Crossref, https://doi.org/10.1109/tpami.2022.3148386.



