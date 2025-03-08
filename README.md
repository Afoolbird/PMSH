# Partially Matching Submap Helps
This repository provides the code for **Partially Matching Submap Helps: Uncertainty Modeling and Propagation
for Text to Point Cloud Localization**

<img src="/overview_1.png"/>

## Introduction
Text to point cloud cross-modal localization is a crucial vision-language task for future human-robot collaboration. Existing coarse-to-fine frameworks assume that each query text precisely corresponds to the center area of a submap, limiting their applicability in real-world scenarios. This work redefines the task under a more realistic assumption, relaxing the one-to-one retrieval constraint by allowing patially matching query text and submap pairs. To address this challenge, we augment datasets with partially matching submaps and introduce an uncertainty-aware framework. Specifically, we model cross-modal ambiguity in fine-grained location regression by integrating uncertainty scores, represented as 2D Gaussian distributions, to mitigate the impact of challenging samples. Additionally, we propose an uncertainty-aware similarity metric that enhances similarity assessment between query text and submaps by propagating uncertainty into coarse place recognition, enabling the model to learn discriminative features, effectively handle partially matching samples and improve task synergy. 


## Installation
Create a conda environment and install basic dependencies:
```bash
conda create -n pmsh python=3.10
conda activate pmsh

# Install the according versions of torch and torchvision
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install required dependencies
CC=/usr/bin/gcc-9 pip install -r requirements.txt
```

## Datasets & Backbone

The KITTI360Pose dataset is used in our implementation.

For training and evaluation, we need cells and poses from Kitti360Pose dataset.
The cells and poses folder can be downlowded from [HERE](https://drive.google.com/file/d/1JT6WALzntau7y_JwYdv5IKJRVPeGzaT0/view?usp=sharing)  

If you want to train the model, you need to download the pretrained object backbone [HERE](https://drive.google.com/file/d/1j2q67tfpVfIbJtC1gOWm7j8zNGhw5J9R/view?usp=drive_link):

```
### Train coarse place recognition

```bash
python -m training.coarse_partial
  --ranking_loss contrastive \
  --hungging_model t5-large \
  --folder_name PATH_TO_COARSE
```

### Train 

```bash
Train fine-grained localization
python -m training.fine_partial --folder_name PATH_TO_FINE
```

## Evaluation

### Evaluation on Val Dataset

```bash
python -m evaluation.pipeline \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```

### Evaluation on Test Dataset

```bash
python -m evaluation.pipeline \
    --use_test_set \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```
