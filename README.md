# AdvML_FinalProject

Neural Network Experiments for Procedural Grid Transformation
This repository contains experiments applying various CNN and Transformer-based architectures to a procedural grid transformation problem. The main goal is to train models to transform input grids into target outputs. The dataset consists of "challenges" and corresponding "solutions", which are split into training and evaluation sets.

Project Overview
The project leverages PyTorch to:

Load and preprocess data:

Convert grids from JSON files into tensors.
Add padding to unify the input sizes.
Group data by task and store it in .pt files.
Model Architectures:
Three primary model architectures are experimented with:

Model1: A baseline CNN + Transformer Encoder architecture.
Model2: A CNN with Inception modules and Residual blocks combined with a Transformer Encoder.
Model3: A CNN with dilated convolutions, residual connections, and a Transformer Encoder.
Training and Evaluation:

Models are trained for a given number of epochs.
Hyperparameter tuning and architecture search is performed using ray[tune].
Multiple configurations are tested for optimal Transformer parameters (e.g., d_model, nhead, num_encoder_layers) and for optimal learning rates and dropout rates.
After training, models are evaluated on a separate evaluation dataset.
Metrics and Visualization:

Track training and evaluation losses and accuracies at each epoch.
Visualize the prediction vs. target grids at various points in training.
Plot and compare the performance curves (e.g., accuracy over epochs) for different models.
Data
The data is downloaded from Google Drive links specified in the code.
Both training (challenges_data.json, solutions_data.json) and evaluation sets (eval_challenges_data.json, eval_solutions_data.json) are processed.
Data is saved into .pt files (grouped_data_by_task.pt), which are then loaded into PyTorch Datasets and DataLoaders.
Models
Model1 (Baseline CNN + Transformer)
Architecture:

A simple CNN stack (3 convolutional layers) followed by batch normalization and ReLU activations.
The feature maps are flattened and fed into a Transformer Encoder.
Outputs are mapped to class predictions for each grid cell.
Best Found Configuration:

d_model=32
nhead=16
num_encoder_layers=3
train_lr=0.01
eval_lr=0.0001
dropout=0.0
Performance:
After training for 10 epochs and evaluating, the model showed stable accuracy improvements.

Model2 (CNN-Inception + Transformer)
Architecture:
Incorporates Inception Modules and Residual Blocks in the CNN portion, adding more representational capacity.
The processed feature maps are fed into a Transformer Encoder.
Best Found Configuration:
d_model=32
nhead=16
num_encoder_layers=2
train_lr=0.1
eval_lr=0.0001
dropout=0.1
Performance:
This model achieved higher training accuracy and better generalization compared to Model1 under certain configurations.

Model3 (Dilated CNN + Transformer)
Architecture:
Uses dilated convolutions to capture larger receptive fields.
Residual connections ensure stable and richer feature extraction.
Features are again passed through a Transformer Encoder.
Best Found Configuration:
d_model=32
nhead=4
num_encoder_layers=2
train_lr=0.001
eval_lr=0.0001
dropout=0.0
Performance:
Demonstrated effective learning with fewer heads and layers, possibly indicating that the dilated and residual connections provided sufficiently rich features for the Transformer to leverage.

Results and Visualizations
Training and evaluation accuracies are plotted to visualize model performance over time. Below are some representative plots:

Model1 Test Evaluation Accuracy:
The accuracy gradually improves over epochs, showing a stable learning curve.

Model2 and Model3 Comparison:
When plotting moving averages of accuracies, Model2 and Model3 may show improved or more stable accuracies compared to Model1, depending on hyperparameter configurations.
