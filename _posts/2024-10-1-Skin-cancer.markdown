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

The **Precision-Recall (PR) curve** is an evaluation metric for binary classification problems, especially useful when dealing with **imbalanced datasets**. It provides insights into the trade-off between two key performance metrics: **precision** and **recall**.

### Key Metrics

1. **Precision (Positive Predictive Value)**: The fraction of relevant instances among the retrieved instances.
   \[
   \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
   \]
   - High precision means fewer false positives.

2. **Recall (Sensitivity or True Positive Rate)**: The fraction of relevant instances that were retrieved.
   \[
   \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
   \]
   - High recall means fewer false negatives.

### Trade-off Between Precision and Recall

- **Precision and recall** are typically inversely related. Increasing recall often reduces precision and vice versa.
- **Threshold tuning** plays a key role in this balance:
  - A **higher threshold** increases precision but decreases recall.
  - A **lower threshold** increases recall but decreases precision.

### PR Curve Plot

The **Precision-Recall curve** is plotted with:
- **X-axis**: Recall (ranging from 0 to 1)
- **Y-axis**: Precision (ranging from 0 to 1)

### Example PR Curve

Below is an example of how precision and recall change with different thresholds:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Simulated true labels and model predicted probabilities
y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.5, 0.3, 0.9, 0.65, 0.2])

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

# Labeling the axes
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Show the plot
plt.legend()
plt.show()

Interpretation
Upper-right region: Represents models with both high precision and high recall, indicating good performance.
Lower-left region: Represents poor performance, with low precision and recall.
A high precision-recall curve means that the model maintains good precision as recall increases.

Applications of PR Curve
Imbalanced Data: PR curves are particularly valuable for datasets with skewed class distributions (e.g., fraud detection, medical diagnosis) where accuracy alone is not sufficient to evaluate the model's performance.
Threshold Selection: By analyzing the PR curve, you can select the best classification threshold that balances precision and recall.
Area Under the Precision-Recall Curve (AUC-PR)
The AUC-PR is another metric derived from the PR curve, which measures the area under the curve. A higher area under the curve signifies better overall performance. Unlike the ROC curve, the PR curve focuses more on the performance of the positive class.

ROC vs. Precision-Recall Curve
ROC Curve: Plots True Positive Rate (Recall) vs False Positive Rate and is more suitable when classes are balanced.
PR Curve: Focuses on the relationship between precision and recall, making it better for imbalanced datasets.


![Graph](/assets/isic_1.jpg)

# ROC AUC Curve Explained

The ROC (Receiver Operating Characteristic) curve and AUC (Area Under the Curve) are essential tools in evaluating the performance of binary classification models. They provide insights into how well a model distinguishes between two classes, typically referred to as positive and negative.

## ROC Curve

- **Definition**: The ROC curve is a graphical representation that illustrates the performance of a binary classifier across various threshold settings. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR).

- **Axes**:
  - **X-axis**: False Positive Rate (FPR), calculated as:
    $$
    \text{FPR} = \frac{FP}{FP + TN}
    $$
  - **Y-axis**: True Positive Rate (TPR), also known as Sensitivity or Recall, calculated as:
    $$
    \text{TPR} = \frac{TP}{TP + FN}
    $$

- The ROC curve is generated by calculating TPR and FPR at various threshold levels. As the classification threshold changes, different pairs of TPR and FPR values are obtained. Lowering the threshold means more items are classified as positive, which increases both TPR and FPR.
-  A curve that hugs the top left corner indicates a high-performing model with a high TPR and low FPR.
- **Interpretation**: Each point on the ROC curve represents a different threshold for classifying instances. A model that perfectly distinguishes between classes will have a point at (0, 1), indicating zero false positives and 100% true positives. Conversely, a model that performs no better than random guessing will lie along the diagonal line from (0, 0) to (1, 1).

## AUC (Area Under the Curve)

- **Definition**: The AUC quantifies the overall performance of the model by measuring the area under the ROC curve.

- **Value Range**:
  - **AUC = 1**: Perfect model; it ranks all positive instances higher than negative ones.
  - **AUC = 0.5**: Model performs no better than random guessing.
  - **AUC < 0.5**: Indicates a model that is worse than random guessing.

- **Interpretation**: A higher AUC value indicates better model performance. For example, an AUC of 0.8 suggests that there is an 80% chance that the model will correctly rank a randomly chosen positive instance higher than a randomly chosen negative instance.

## Practical Applications

1. **Model Comparison**: AUC is particularly useful for comparing different models. The model with the higher AUC is generally considered superior.
  
2. **Threshold Selection**: The ROC curve helps in selecting an optimal threshold based on the trade-off between TPR and FPR, depending on the specific requirements of the application.

3. **Imbalanced Datasets**: While ROC and AUC are effective for balanced datasets, they can be misleading in imbalanced scenarios. In such cases, precision-recall curves may provide better insights.

## Summary

The ROC curve and AUC are powerful metrics for evaluating binary classifiers. They provide a comprehensive view of a model's ability to distinguish between classes across various thresholds, making them invaluable tools in machine learning and statistical analysis. By analyzing these metrics, practitioners can make informed decisions about model selection and threshold optimization to meet specific objectives in their classification tasks.

# Partial ROC Curve

## Introduction

The **Receiver Operating Characteristic (ROC) Curve** is a widely-used tool to evaluate the performance of binary classification models by plotting the trade-off between **True Positive Rate (Recall)** and **False Positive Rate**. However, in many real-world scenarios, we are interested in a specific region of the ROC curve, particularly when the **false positive rate** is constrained to low values. This is where the **Partial ROC Curve (pROC)** comes into play.

### ROC Curve Overview

- **True Positive Rate (TPR)**, also known as **Recall**:
  \[
  TPR = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  \]
  
- **False Positive Rate (FPR)**:
  \[
  FPR = \frac{\text{False Positives (FP)}}{\text{False Positives (FP)} + \text{True Negatives (TN)}}
  \]

- The **ROC Curve** plots TPR against FPR at various classification thresholds. The curve shows how the model’s sensitivity and false positive rate change as the decision threshold is varied.

### Why Partial ROC Curve?

In many applications, it is not practical to examine the entire ROC curve. For example, in medical diagnosis or fraud detection, a high **False Positive Rate** is undesirable, and we may only be interested in how the model performs within a low FPR range.

The **Partial ROC Curve** (pROC) focuses on a **subset** of the ROC curve, typically when the FPR is below a certain threshold, say 0.1 (i.e., 10%). This allows for better evaluation of the model’s performance in the region that matters most.

### Applications of Partial ROC

- **Medical Diagnosis**: In situations where false positives lead to costly or harmful interventions, we may want to analyze the model performance only at very low FPR values (e.g., less than 5%).
  
- **Fraud Detection**: A high number of false positives can result in unnecessary investigations, so it is critical to examine model performance at low false positive rates.

- **Anomaly Detection**: Many anomaly detection systems are designed to operate at low FPRs to minimize the rate of false alarms.

### Plotting the Partial ROC Curve

The process of generating a partial ROC curve is similar to the full ROC curve, except we restrict the x-axis to a lower range of FPR values (e.g., between 0 and 0.1).

### Example Python Code for Partial ROC Curve

Here’s an example of how to compute and plot a **Partial ROC Curve** in Python using `scikit-learn`:

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Simulated true labels and model predicted probabilities
y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.5, 0.3, 0.9, 0.65, 0.2])

# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Define the range of FPR we are interested in (partial)
max_fpr = 0.1
partial_fpr = fpr[fpr <= max_fpr]
partial_tpr = tpr[:len(partial_fpr)]

# Calculate partial AUC (optional)
partial_auc = auc(partial_fpr, partial_tpr)

# Plot the partial ROC curve
plt.figure(figsize=(8, 6))
plt.plot(partial_fpr, partial_tpr, marker='.', label=f'Partial ROC (AUC = {partial_auc:.2f})')

# Labeling the axes
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Partial ROC Curve')

# Show the plot
plt.legend()
plt.show()```
Partial Area Under the ROC Curve (pAUC)
Similar to the full Area Under the ROC Curve (AUC-ROC), we can compute the Partial AUC (pAUC), which measures the area under the partial ROC curve. The pAUC is particularly important when evaluating models in a restricted FPR range, as it summarizes the model’s performance in that region.

For example, if we are only concerned with an FPR range from 0 to 0.1, the pAUC will provide a more relevant metric than the full AUC-ROC.

Advantages of Partial ROC Curve
Focus on Relevant Performance: The full ROC curve often includes regions that are not of practical interest, especially when FPR values are high. The pROC curve allows for targeted evaluation within specific FPR ranges that are critical for the application.

Better Comparison: In domains like medical diagnosis, comparing models based on their pROC performance (e.g., within FPR < 0.1) gives a more meaningful insight than considering the full ROC curve.

Improved Decision-Making: By focusing on a limited FPR range, stakeholders can make better decisions about which model to deploy in critical, high-stakes environments.

Limitations
Choice of FPR Threshold: The choice of the maximum FPR value for the partial ROC curve can be somewhat arbitrary and may need to be tuned based on the specific domain or use case.

Loss of Full Picture: By focusing on a partial region of the ROC curve, we may lose valuable insights into the model’s performance at other FPR values.

Summary
The Partial ROC Curve provides a focused view of a classifier’s performance in the region of interest, where controlling the False Positive Rate is critical. This tool is valuable in scenarios where only a low false positive rate is acceptable, allowing for better model evaluation and selection in high-stakes applications like medical diagnosis, fraud detection, and anomaly detection.