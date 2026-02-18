# Satellite Image Classification using CNN

A comprehensive deep learning project that implements and compares multiple Convolutional Neural Network (CNN) architectures for satellite image classification. This project demonstrates the impact of data augmentation, normalization, and architectural design choices on model performance.

## Problem Description and Motivation

Satellite imagery analysis is crucial for Earth observation, urban planning, environmental monitoring, and disaster response. Automated classification of satellite images into land cover categories enables rapid assessment of geographic regions and resource allocation optimization.

**Key Objectives:**
- Develop accurate CNN models for multi-class satellite image classification
- Compare baseline and augmented model architectures to understand the impact of data augmentation
- Evaluate model generalization and robustness on unseen test data
- Provide insights into feature learning through convolutional filter visualization

## Dataset Description

### Source
- **Dataset Name:** Satellite Image Classification
- **Source:** [Kaggle - mahmoudreda55/satellite-image-classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)
- **Download Method:** KaggleHub Python API

### Dataset Specifications
- **Image Size:** 64 × 64 pixels (RGB color images)
- **Classes:** Multiple land cover categories (urban, green areas, water, etc.)
- **Data Split:**
  - Training Set: 80% of unique images
  - Validation Set: 10% of unique images
  - Test Set: 10% of unique images
- **Preprocessing:** Duplicate image removal using MD5 hash validation

## Setup and Running Instructions

### Prerequisites
- Python 3.8 or higher
- macOS with Apple Silicon (M1/M2) or alternative CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mwpersson/mini-project-5.git
   cd mini-project-5
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify TensorFlow and GPU setup:**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
   ```

### Running the Notebook

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/cnn_classifier.ipynb
   ```

2. **Execute cells sequentially:**
   - Cell 1: Setup and library imports (verify TensorFlow and GPU availability)
   - Cell 2-3: Dataset download and preprocessing
   - Cells 4-6: Data exploration and pipeline optimization
   - Cells 7-16: Model training and evaluation

### Dependencies
See `requirements.txt` for complete list:
- TensorFlow/Keras (with Metal acceleration for macOS)
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- KaggleHub

## Results Summary

### Key Metrics

| Metric | Baseline Model | Augmented Model | Experimental Model |
|--------|---|---|---|
| **Train Accuracy** | 98.71% | 98.86% | N/A |
| **Validation Accuracy** | 98.83% | 80.66% | N/A |
| **Test Accuracy** | 99.02% | 82.81% | N/A |
| **Test Loss** | 0.0304 | 1.0311 | N/A |
| **Training Epochs** | 20 | 20 | 60 |

### Model Architectures

**Baseline Model (model_basic):**
- 3 Convolutional blocks (32, 64, 128 filters)
- MaxPooling after each Conv block
- 2 Dense layers (128 units, softmax output)
- **Parameters:** ~340K
- **Best Test Accuracy:** 99.02%

**Augmented Model (model_augmented):**
- Input normalization (rescaling to [0, 1])
- Data augmentation layers (random flip, rotation, zoom)
- 3 Convolutional blocks with BatchNormalization
- MaxPooling and Dropout (0.3)
- 2 Dense layers
- **Parameters:** ~350K
- **Best Test Accuracy:** 82.81%

**Experimental Model (model_experimental):**
- Similar architecture with GlobalAveragePooling2D
- AdamW optimizer
- Extended training (60 epochs)
- Tested for longer convergence

### Key Findings

1. **Baseline Dominance:** The baseline model without augmentation achieved a good test accuracy (99.02%), indicating the satellite images are relatively distinct and well-separated in feature space.

2. **Augmentation Trade-off:** While data augmentation is typically beneficial, it degraded performance in this project, suggesting:
   - The dataset is sufficiently diverse without augmentation
   - Strong augmentation (rotation, zoom) may disrupt spatial relationships critical for satellite classification
   - The baseline model achieved low generalization gap (Val Acc: 98.83%), indicating sufficient regularization

3. **Validation Gap:** The augmented model shows significant overfitting (Val Acc: 80.66% vs Test Acc: 82.81%), despite dropout regularization.

4. **Model Efficiency:** All models achieved strong accuracy with similar parameter counts (~340K), demonstrating efficient architecture design.

### Training Curves

The training curves show:
- **Baseline:** Smooth convergence with minimal train-val gap
- **Augmented:** Faster initial learning but higher variance in validation performance
- **Loss Stability:** Both models achieve low loss values, with baseline showing more stable optimization

The notebook includes comprehensive visualization of model predictions across test samples:

- **Green correct predictions** indicate accurate classifications
- **Red incorrect predictions** highlight misclassified samples
- **Confidence scores** show model certainty for each prediction

### Feature Map Visualization

Convolutional filter activations are visualized for the baseline model, showing:
- Early layers learn edge and texture detection
- Middle layers identify intermediate patterns (land boundaries, water bodies)
- Late layers capture semantic features (building clusters, agricultural patches)

## Team Member Contributions

This project was developed as a comprehensive machine learning demonstration:

- **Data Acquisition & Preprocessing:** Dataset collection, duplicate removal, train-validation-test split
- **Exploratory Data Analysis:** Class distribution analysis, sample visualization, batch inspection
- **Model Architecture Design:** Three distinct CNN architectures with varying regularization strategies
- **Training & Evaluation:** Training pipelines with performance monitoring, comprehensive metric computation
- **Visualization & Analysis:** Training curves, confusion matrices, feature map visualization, detailed comparisons
- **Documentation:** Model specifications, results analysis, findings interpretation

## Future Improvements

1. **Architecture Enhancements:**
   - ResNet/DenseNet transfer learning
   - Attention mechanisms for spatial focus
   - Multi-scale feature extraction

2. **Data Strategy:**
   - Class-balanced augmentation
   - Mixup/CutMix regularization
   - Synthetic oversampling of minority classes

3. **Optimization:**
   - Learning rate scheduling and warmup
   - Advanced optimizers (RAdam, LAMB)
   - Mixed precision training


## References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs
- Satellite Imagery ML: https://arxiv.org/abs/1505.04597
- Deep Learning Best Practices: https://cs231n.github.io/


