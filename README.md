# Hamster Expression Recognition 🐹

A from-scratch PCA-based facial expression recognition system that uses webcam input to classify facial expressions and maps each prediction to a humorous hamster reaction image.

## Overview

Hamster Expression Recognition is a computer vision and machine learning project built to explore facial expression recognition using mathematical techniques rather than pre-built recognition libraries.

The system captures a face through a webcam, preprocesses it into a grayscale image, projects it into a reduced-dimensional PCA space, and classifies the expression using class centroids.

The predicted expression is then mapped to a corresponding hamster reaction image.

### Supported Expressions

- Angry
- Cry
- Happy
- Neutral
- Sad
- Surprised

## How It Works

The recognition pipeline follows these steps:

```text
Webcam
   ↓
Face Detection
   ↓
Face Cropping & Preprocessing
   ↓
64 × 64 Grayscale Image
   ↓
Mean Centering
   ↓
PCA
   ↓
Eigenfaces
   ↓
Projection into PCA Space
   ↓
Weighted Centroid Classification
   ↓
Expression Prediction
   ↓
Hamster Reaction 🐹