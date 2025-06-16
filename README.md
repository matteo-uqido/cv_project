# Crowd Counting Models Project

Welcome to the **Crowd Counting Models Project**, a university project focused on implementing and training various deep learning models for crowd counting tasks. This repository contains modularized code for model creation, training, and evaluation, along with configuration files for fine-tuning hyperparameters.

---

## Project Overview

Crowd counting is a computer vision task that estimates the number of people in an image. This project explores multiple state-of-the-art architectures, including **ResNet50**, **VGG16**, **VGG19**, **Xception**, and **CSRNet**, to tackle this problem.

The repository is structured to ensure **scalability**, **modularity**, and **ease of experimentation**.

---

## Features

- **Model Creation**: Dynamically create models based on user input (`resnet50`, `vgg16`, `vgg19`, `xception`, `csrnet`).
- **Training Pipeline**: Train models with configurable hyperparameters using **Keras** and **TensorFlow**.
- **Loss Functions**: Support for custom loss functions like **Euclidean loss** and **Mean Squared Error (MSE)**.
- **Callbacks**: Integrated callbacks for **early stopping** and **learning rate reduction**.
- **Configuration Management**: **YAML-based** configuration files for easy parameter tuning.

---

## Repository Structure

```bash
cv_project/
├── config/
│   ├── models_parameters.yaml       # Hyperparameters for each model
│   └── config_loader.py             # Utility for loading configurations
├── modules/
│   ├── models/
│   │   ├── create_desired_model.py  # Factory for model creation
│   │   ├── CSRNET_model.py          # CSRNet implementation
│   │   └── crowd_counting_models.py # ResNet50, VGG16, VGG19, Xception implementations
│   ├── training_eval_pipeline/
│   │   ├── training_functions.py    # Training and evaluation logic
│   │   └── evaluation_functions.py  # Evaluation metrics and utilities
└── README.md                        # Project documentation
``` 


# Project documentation

---

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Keras
- PyYAML

---

## Configuration File

The `models_parameters.yaml` file contains hyperparameters for each model. Example configuration:

```yaml
resnet50:
  parameters:
    adam_lr: 0.001
    epochs: 100
    loss: euclidean
    metrics: ['mse']
    es_patience: 5
    monitor: val_mse
    es_min_delta: 0.001
```

