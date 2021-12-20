# AirwaySymptomDetection

This repository contains related code to the paper below.
All relevant data is available on [Scripted Rainbow Passage](https://faubox.rrze.uni-erlangen.de/getlink/fi2qBUcwgWyZaQaWqJEqBf3i/Old_Data) and [Lei et al., 2020](https://faubox.rrze.uni-erlangen.de/getlink/fiVW7DbcUiHkJVyAsK3nq47Q/New_Data).
Data from the COUGHVID crowdsourcing dataset is available on [zenodo](https://zenodo.org/record/4048312#.YcCYJseZNnI).

## How to use the code

To use the code, you need a Python installation together with relevant libraries (librosa, numpy, scikit-learn, tensorflow).

## Training deep neural networks

We provide code to train several deep neural network architectures (`neural_networks`), e.g., ResNet, EfficientNet or RNNs. In `analysis`, you find a Jupyter notebook for evaluating the trained models.
Aditionally, the repository provides code for training an autoencoder architecture (`neural_networks`) for converting from microphone-acoustic to mechano-acoustic Mel-spectrograms.

## Genetic algorithm

We provide code to run a genetic algorithm (`neural_networks/GeneticAlgorithm`) to optimize and find a low-size, accurate neural network architecture. 

## Evaluate Genetic Algorithm model's performance on more unseen data

In `analysis`, you find two Jupyter notebooks for evaluating the Objective 2 model, which was determined using the Genetic Algorithm. First dataset: Lei et al., 2020 - `predict-new-dataset-mcgill.ipynb`, second dataset: COUGHVID crowdsourcing dataset - `predict-cough-database.ipynb`.

## Explainable AI

In `analysis`, you find Jupyter notebooks for visualizing the class activation maps and results from occlusion experiments.

## How to cite this code

Groh et al. "Efficient and Explainable Deep Neural Networks for Airway Symptom Detection in Support of Wearable Health Technology", 2021
