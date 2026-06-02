# CV Transfer Learning & Error Analysis

A personal Computer Vision mini-project using a pretrained ResNet18 model for ants-vs-bees image classification, completed as part of my preparation for an AI internship.

## Project Overview

This project explores transfer learning for image classification using a pretrained ResNet18 model. Two approaches were evaluated:

1. Fixed feature extractor: freeze the pretrained backbone and train only a new classification head.
2. Partial fine-tuning: unfreeze the final ResNet block (`layer4`) together with the classification head.

The project also includes confusion matrix analysis and Grad-CAM visualizations to inspect high-confidence misclassifications.

## Key Results

| Model | Validation Accuracy | Confusion Matrix |
|---|---:|---|
| Fixed Feature Extractor | 96.08% | `[[68, 2], [4, 79]]` |
| Partial Fine-Tuning | 96.08% | `[[67, 3], [3, 80]]` |

## Main Findings

- Training only the new classification head achieved strong validation performance with a small public dataset.
- Grad-CAM suggested possible context bias in several errors, such as ants on flowers being predicted as bees.
- Some brown or unusual-looking bees were attended to correctly but still classified as ants, suggesting object-level confusion.
- Partial fine-tuning redistributed class-specific errors but did not improve overall validation accuracy.
- The fixed-feature baseline was selected as the final model because it achieved the same accuracy with a simpler training setup.

## Technical Stack

- Python
- PyTorch / Torchvision
- ResNet18 Transfer Learning
- Scikit-learn
- Matplotlib
- Grad-CAM
- Google Colab

## Repository Structure

```text
.
├── cv_model_01_transfer_learning.ipynb
└── results/
    ├── baseline_confusion_matrix.png
    ├── partial_confusion_matrix.png
    ├── baseline_accuracy_curve.png
    ├── baseline_loss_curve.png
    ├── partial_accuracy_curve.png
    ├── partial_loss_curve.png
    ├── baseline_vs_partial_comparison.csv
    └── gradcam_*.png
