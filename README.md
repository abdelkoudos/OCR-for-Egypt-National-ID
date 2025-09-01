# OCR for Egyptian National ID

This repository contains three projects that focus on building end-to-end OCR pipelines for Egyptian National ID cards. Each project addresses a different component of the ID, combining computer vision, image preprocessing, segmentation, and deep learning models to achieve high recognition accuracy.  

## Projects Overview

### 1. Categorical Field Classification
- Classifies **gender, religion, and marital status** from segmented ID fields.  
- Uses separate CNN models for each field, addressing **class imbalance** with data augmentation and synthetic data generation.  
- Achieves approximatly 99% F1 scores across all components.  

### 2. Serial Number Recognition
- Extracts and recognizes the **ID serial number** (two letters + seven digits).  
- Employs **wavelet-based denoising, adaptive thresholding, and contour analysis** for segmentation.  
- Custom CNN models outperform pretrained OCR tools (Keras OCR, Tesseract) with >93% overall accuracy.  

### 3. Back ID 14-Digit Recognition
- End-to-end system for recognizing the **14-digit number** on the back of the ID.  
- Pipeline includes robust preprocessing, contour-based segmentation, and CNN classification.  
- Achieves >94% line-level accuracy after iterative dataset cleaning and retraining.  

## Notes
- Each project is located in its own folder with a dedicated README containing detailed methodology, experiments, and results.  
- Datasets used in these projects are **confidential** and cannot be shared.  
