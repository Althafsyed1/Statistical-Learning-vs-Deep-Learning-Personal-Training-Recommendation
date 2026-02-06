# Statistical-Learning-vs-Deep-Learning-Personal-Training-Recommendation
# 🏋️ Statistical Learning vs Deep Learning  
### Personal Training Recommendation & Exercise Classification

[![Domain](https://img.shields.io/badge/Domain-Machine%20Learning-blue.svg)]()
[![Focus](https://img.shields.io/badge/Focus-Model%20Benchmarking-success.svg)]()
[![Data](https://img.shields.io/badge/Data-Tabular%20%2B%20Images-orange.svg)]()
[![Validation](https://img.shields.io/badge/Validation-Report%20%2B%20Experiments-brightgreen.svg)]()

> **A comparative machine learning study evaluating Statistical Learning and Deep Learning approaches for personal fitness recommendation and exercise classification.**  
>  
> This project focuses on **model selection trade-offs, interpretability, accuracy, inference speed, and deployment cost**, supported by **experimental benchmarking and a detailed academic report (PDF)**.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Datasets](#-datasets)
- [System Architecture](#-system-architecture)
- [Statistical Learning Approach](#-statistical-learning-approach)
- [Deep Learning Approach](#-deep-learning-approach)
- [Experimental Validation](#-experimental-validation)
- [Results & Observations](#-results--observations)
- [Key Insights](#-key-insights)
- [Challenges & Considerations](#-challenges--considerations)
- [Future Enhancements](#-future-enhancements)
- [Repository Contents](#-repository-contents)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

### Motivation
Modern fitness and personal training systems increasingly rely on machine learning. However, **higher accuracy does not always imply better real-world suitability**.

Key questions addressed in this project:
- When are **interpretable statistical models sufficient**?
- When is **deep learning necessary**?
- What are the **trade-offs in accuracy, speed, memory, and explainability**?

This project provides **quantitative and qualitative evidence** to guide model selection for real-world fitness applications.

---

## ❓ Problem Statement

Fitness recommendation systems must operate under:
- Limited compute resources (mobile / embedded devices)
- Need for explainability and transparency
- Real-time inference constraints
- Mixed data types (tabular + visual)

A single modeling approach is often insufficient.

---

## 🎯 Project Objectives

- Compare **Statistical Learning vs Deep Learning** on identical tasks
- Benchmark models using **realistic datasets**
- Evaluate:
  - Accuracy
  - Interpretability
  - Training cost
  - Inference latency
  - Model size
- Provide **deployment-oriented insights**, not just accuracy numbers

---

## 📊 Datasets

### 1️⃣ Structured Exercise Dataset
- **2,918 exercise records**
- Features include:
  - Body part
  - Equipment
  - Difficulty
  - User ratings
- Used for:
  - Exercise recommendation
  - Difficulty classification

---

### 2️⃣ Image-Based Exercise Dataset
- **13,853 labeled images**
- **22 exercise classes**
- Used for:
  - Vision-based exercise classification
  - Deep learning benchmarking

---


---

## 📐 Statistical Learning Approach

### Models Implemented
- Decision Tree (Entropy criterion)
- Logistic Regression (L2 regularization)

### Key Techniques
- One-hot encoding
- Feature normalization
- **Random Forest–based rating imputation**
  - RMSE ≈ **1.761**
- ROC–AUC evaluation
- Confusion matrix analysis

### Design Focus
- Interpretability
- Low inference latency
- Minimal memory footprint
- Explainable decision paths

---

## 🧠 Deep Learning Approach

### Architectures Used
- MobileNetV2
- VGG16
- ResNet (ImageNet-pretrained)

### Training Strategy
- Transfer learning
- Data augmentation (rotation, zoom, shifts)
- Frozen backbone with custom classifier head
- GPU-accelerated training

### Design Focus
- High accuracy on unstructured image data
- Robust feature extraction
- Scalability to large datasets

---
## 📂 Dataset Access

Due to GitHub file size limitations, the image dataset used for deep learning
experiments (~13,853 images, 22 classes) is hosted externally.

📥 **Download link:**  
👉 https://drive.google.com/file/d/1i32In2O5TrpmGGjVvJHEvnz6-8SD9vgu/view?usp=sharing

The dataset is provided in ZIP format and should be extracted into:

data/images/


## 🧪 Experimental Validation

Validation is performed through:
- Controlled train/test splits
- ROC–AUC analysis
- Precision, Recall, F1-score
- Training time measurement
- Inference latency benchmarking
- Model size comparison

📄 All experimental details and results are documented in the **PDF report included in this repository**.

---

## 📊 Results & Observations

| Model | Test Accuracy | Training Time | Inference Time | Model Size |
|------|---------------|---------------|---------------|-----------|
| Decision Tree | **84.6%** | ~7 s | **~0.01 s** | < 1 MB |
| Logistic Regression | 72.9% | ~2 s | ~0.03 s | < 1 MB |
| CNN (MobileNetV2) | **95%** | ~1200 s | ~0.5 s / 1000 imgs | ~120 MB |

---

## 🔍 Key Insights

- Statistical Learning:
  - Strong performance on structured data
  - Highly interpretable
  - Ideal for real-time and embedded systems
- Deep Learning:
  - Superior accuracy for image-based tasks
  - Higher computational and memory cost
  - Limited interpretability
- **Hybrid systems** provide the best real-world balance

---

## ⚠️ Challenges & Considerations

- Deep learning models incur high training cost
- Statistical models struggle with unstructured data
- Deployment constraints often outweigh accuracy gains
- Interpretability remains critical in recommendation systems

---

## 🔮 Future Enhancements

- Hybrid SL + DL recommendation pipeline
- Real-time deployment optimization
- User-specific personalization models
- Lightweight CNN inference optimization
- Mobile or edge-device deployment

---


---

## 📄 License
Academic and research use only.

---

## 👤 Author

**Mohammad Althaf Syed**  
**Anthony Huang**

Stevens Institute of Technology

[GitHub](https://github.com/Althafsyed1)  
[LinkedIn](https://linkedin.com/in/yourprofile)

---

<div align="center">

### ⭐ If this project was useful, please consider starring the repository

**Machine Learning • Model Benchmarking • Deployment Trade-offs** 🧠📊

</div>

---

**Project Type:** Comparative Study  
**Primary Tools:** Python, Scikit-learn, TensorFlow  
**Validation:** Experimental Results + PDF Report  
**Status:** Complete & Ready for Extension



