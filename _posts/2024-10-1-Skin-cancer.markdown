---
layout: post
title: "Skin Cancer"
date: 2024-10-01 01:43:18 +0530
categories: kaggle
---

# ISIC 2024 - Skin Cancer Detection with 3D-TBP 
Skin cancer can be deadly if not caught early, but many populations lack specialized dermatologic care. Over the past several years, dermoscopy-based AI algorithms have been shown to benefit clinicians in diagnosing melanoma, basal cell, and squamous cell carcinoma. However, determining which individuals should see a clinician in the first place has great potential impact. Triaging applications have a significant potential to benefit underserved populations and improve early skin cancer detection, the key factor in long-term patient outcomes.

In this project, I aimed to develop a binary classification algorithm to identify skin cancer from single-lesion crops of 3D total body photos (TBP). Skin cancer, if not detected early, can be life-threatening, especially in underserved populations lacking access to specialized dermatologic care. Given that telehealth submissions often involve lower-quality images, similar to smartphone photos, my goal was to create an algorithm that works in these non-clinical settings.

By building this project, I sought to extend the benefits of AI-based skin cancer detection to broader populations, enhancing early diagnosis and triage in resource-constrained environments.

In this competition, we  develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). The image quality resembles close-up smartphone photos, which are regularly submitted for telehealth purposes. Our binary classification algorithm could be used in settings without access to specialized care and improve triage for early skin cancer detection.

# Evaluation 

![table](/assets/isic_2.jpg)

# Confusion Matrix Example

A confusion matrix is a useful tool for evaluating the performance of a classification model.

## Confusion Matrix

|                | Predicted Positive | Predicted Negative |
|----------------|---------------------|---------------------|
| **Actual Positive** | True Positive (TP)  | False Negative (FN)  |
| **Actual Negative** | False Positive (FP) | True Negative (TN)   |

### Explanation of Terms

- **True Positive (TP)**: The number of positive instances correctly predicted as positive.
- **False Negative (FN)**: The number of positive instances incorrectly predicted as negative.
- **False Positive (FP)**: The number of negative instances incorrectly predicted as positive.
- **True Negative (TN)**: The number of negative instances correctly predicted as negative.

### Example Values

To give you a clearer picture, here's an example with hypothetical values:

|                | Predicted Positive | Predicted Negative |
|----------------|---------------------|---------------------|
| **Actual Positive** | 50                  | 10                  |
| **Actual Negative** | 5                   | 100                 |

### Performance Metrics

From this confusion matrix, you can derive several performance metrics:

- **Accuracy**: $$(TP + TN) / (TP + TN + FP + FN)$$
- **Precision**: $$TP / (TP + FP)$$
- **Recall (Sensitivity)**: $$TP / (TP + FN)$$
- **F1 Score**: $$2 \times \frac{Precision \times Recall}{Precision + Recall}$$

Accuracy is a widely used metric in machine learning, but it can be misleading in many contexts:
- Imbalance in Data (Class Imbalance Problem)
Description: Accuracy is unreliable when dealing with imbalanced datasets, where one class is significantly more frequent than others.
Example: Consider a dataset with 95% of class A and 5% of class B. A model that predicts class A for every instance would still achieve an accuracy of 95%, even though it's failing to detect class B. This would give a false sense of model performance.
Better Metrics: Precision, recall, F1-score, or area under the ROC curve (AUC-ROC) provide more insight into the performance of models dealing with imbalanced data.
- Inability to Capture the Severity of Errors
Description: Accuracy does not distinguish between different types of errors or their consequences. It treats all incorrect predictions the same, regardless of the severity.
Example: In medical diagnosis, predicting that a patient is healthy when they have a life-threatening disease (false negative) is far more critical than wrongly predicting a healthy person as ill (false positive). Accuracy would not capture this difference.
Better Metrics: In such cases, you can use weighted accuracy or specific metrics like sensitivity (recall) and specificity to reflect the consequences of false negatives and false positives.
- Lack of Sensitivity to Data Distribution
Description: Accuracy does not account for the distribution of classes or categories. It assumes that all classes are equally important or that misclassification has the same impact across the dataset.
Example: If you're predicting fraud in transactions, where 1% of the data represents fraudulent transactions and 99% represents legitimate transactions, a high accuracy may just reflect the model's ability to predict the majority class (legitimate transactions), but not fraud.
Better Metrics: You can use precision-recall curves or confusion matrices to get more granular insights into how the model performs on each class, especially the minority class.


# TPR and FPR

## True Positive Rate (TPR)
- **Definition**: Proportion of actual positives correctly identified.
- **Formula**: 
  $$
  \text{TPR} = \frac{TP}{TP + FN}
  $$

## False Positive Rate (FPR)
- **Definition**: Proportion of actual negatives incorrectly identified as positives.
- **Formula**: 
  $$
  \text{FPR} = \frac{FP}{FP + TN}
  $$

### Summary Table

| Metric | Formula |
|--------|---------|
| TPR    | $$\frac{TP}{TP + FN}$$ |
| FPR    | $$\frac{FP}{FP + TN}$$ |

# Precision-Recall Curve

## Introduction

The **Precision-Recall (PR) curve** is an essential evaluation tool for **binary classification problems**, particularly when dealing with **imbalanced datasets**. It illustrates the trade-off between two critical metrics: **precision** and **recall**, helping assess the performance of a model, especially in scenarios where the positive class is rare.

### Key Metrics

1. **Precision (Positive Predictive Value)**: Measures the proportion of true positive predictions among all positive predictions.
   \[
   \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
   \]
   - A high precision score indicates a low number of false positives.

2. **Recall (Sensitivity or True Positive Rate)**: Reflects the proportion of actual positive instances that are correctly identified.
   \[
   \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
   \]
   - A high recall means that most positive cases are detected by the model.

### Trade-off Between Precision and Recall

- **Precision and recall** typically have an inverse relationship:
  - Increasing the threshold for classifying a positive sample often results in higher precision but lower recall (fewer false positives but also fewer true positives).
  - Lowering the threshold increases recall but may reduce precision, as more false positives are included.
- **Threshold tuning** is crucial to balance these metrics based on the specific goals of the model.

### PR Curve Plot

The **Precision-Recall curve** visualizes how precision and recall vary at different classification thresholds:
- **X-axis**: Recall (ranging from 0 to 1).
- **Y-axis**: Precision (ranging from 0 to 1).

In the ideal case, precision remains high as recall increases, reflecting a strong model.

### Example PR Curve

Here's an example code to generate a Precision-Recall curve using Python:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Simulated true labels and predicted probabilities
y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.5, 0.3, 0.9, 0.65, 0.2])

# Calculate precision, recall, and thresholds
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Plot the precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

### Interpretation

- **Upper-right region**: Indicates both high precision and high recall, showing good performance.
- **Lower-left region**: Indicates low precision and recall, representing poor model performance.
- A curve closer to the upper-right corner represents a more effective model.

### Applications of PR Curve

- **Imbalanced Datasets**: The PR curve is particularly valuable for evaluating models in situations where the positive class is underrepresented, such as fraud detection or medical diagnosis.
- **Threshold Selection**: By analyzing the PR curve, one can choose a classification threshold that best balances precision and recall according to the problem at hand.

### Area Under the Precision-Recall Curve (AUC-PR)

The **AUC-PR** (Area Under the Precision-Recall Curve) is a single-value metric derived from the PR curve. It provides an overall measure of the model's performance:
- A **higher AUC-PR** value indicates a better ability to maintain precision as recall increases.

### ROC vs. Precision-Recall Curve

- **ROC Curve**: Focuses on the **True Positive Rate** vs **False Positive Rate**, and is better suited when class distributions are relatively balanced.
- **PR Curve**: Focuses on the trade-off between precision and recall, making it more appropriate for **imbalanced datasets**.

---

# Partial ROC Curve

## Introduction

The **Partial ROC (pROC) curve** focuses on evaluating the performance of a binary classification model in a specific region of the **ROC curve**, particularly when the **False Positive Rate (FPR)** is constrained to low values. This is especially useful in domains where controlling the false positive rate is crucial, such as medical diagnosis or fraud detection.

### ROC Curve Overview

- **True Positive Rate (TPR)**, or **Recall**:
  \[
  TPR = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  \]
  
- **False Positive Rate (FPR)**:
  \[
  FPR = \frac{\text{False Positives (FP)}}{\text{False Positives (FP)} + \text{True Negatives (TN)}}
  \]

The **ROC curve** plots TPR vs FPR at different classification thresholds. A good model has a curve that hugs the top-left corner (high TPR, low FPR).

### Why Partial ROC?

In certain applications, we are only concerned with the model's performance at **low FPR**. For instance:
- **Medical Diagnosis**: A low FPR is critical to avoid unnecessary and costly treatments.
- **Fraud Detection**: Minimizing false positives is important to reduce false alarms and investigations.

The **Partial ROC Curve** focuses on a subset of the ROC curve, usually up to a predefined FPR threshold (e.g., 0.1 or 0.2).

### Plotting the Partial ROC Curve

Here’s how to compute and plot a **Partial ROC Curve** in Python:

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Simulated true labels and predicted probabilities
y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.5, 0.3, 0.9, 0.65, 0.2])

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Define the maximum FPR for the partial ROC
max_fpr = 0.1
partial_fpr = fpr[fpr <= max_fpr]
partial_tpr = tpr[:len(partial_fpr)]

# Calculate partial AUC (optional)
partial_auc = auc(partial_fpr, partial_tpr)

# Plot the partial ROC curve
plt.plot(partial_fpr, partial_tpr, marker='.', label=f'Partial ROC (AUC = {partial_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Partial ROC Curve')
plt.legend()
plt.show()
```

### Partial Area Under the ROC Curve (pAUC)

Just like the full AUC, the **Partial AUC (pAUC)** is the area under the partial ROC curve. It summarizes the model's performance within the restricted FPR range, providing a more focused evaluation.

### Advantages of Partial ROC

1. **Relevance**: The partial ROC curve allows for targeted evaluation within specific **FPR ranges**, which are often critical in applications like healthcare or security.
2. **Better Decision-Making**: By focusing on low FPR values, stakeholders can select models that reduce the risks of costly false positives.

### Limitations

- **Choice of FPR Threshold**: Selecting the maximum FPR value for the partial ROC curve may be somewhat arbitrary and needs to be aligned with the specific application.
- **Loss of Global Insight**: Focusing on a small region of the ROC curve might lead to missing insights about the model's performance in other FPR regions.

---

## Summary

Both the **Precision-Recall Curve** and the **Partial ROC Curve** are powerful tools for evaluating classifiers, particularly in domains where **class imbalance** or **false positive control** is critical. Each provides a focused view of model performance, allowing for better decision-making and model selection in high-stakes environments.