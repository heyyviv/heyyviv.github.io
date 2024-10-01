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



![table](/assets/isic_1.jpg)

## Precision-Recall Curve

### Introduction

The **Precision-Recall (PR) curve** is an evaluation metric for binary classification problems, particularly useful when dealing with **imbalanced datasets**. It provides insights into the trade-off between two key performance metrics: **precision** and **recall**.

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

- **Precision and recall** are inversely related. Increasing recall often reduces precision and vice versa.
- **Threshold tuning** plays a key role in this balance:
  - A **higher threshold** increases precision but decreases recall.
  - A **lower threshold** increases recall but decreases precision.

### PR Curve Plot

The **Precision-Recall curve** is plotted with:

- **X-axis**: Recall (ranging from 0 to 1)
- **Y-axis**: Precision (ranging from 0 to 1)

### Example PR Curve

Here’s a Python example to generate a Precision-Recall curve:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
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
```

### Interpretation

- **Upper-right region**: Represents models with both high precision and high recall, indicating good performance.
- **Lower-left region**: Represents poor performance, with low precision and recall.
- A high precision-recall curve means that the model maintains good precision as recall increases.

### Applications of PR Curve

- **Imbalanced Data**: PR curves are particularly valuable for datasets with skewed class distributions (e.g., fraud detection, medical diagnosis).
- **Threshold Selection**: By analyzing the PR curve, you can select the best classification threshold that balances precision and recall.

### Area Under the Precision-Recall Curve (AUC-PR)

The **AUC-PR** measures the area under the PR curve, indicating overall model performance. A higher area signifies better performance, particularly focusing on the positive class.

---

## ROC Curve and AUC

### ROC Curve Overview

The **Receiver Operating Characteristic (ROC) curve** is used to evaluate the performance of binary classification models by plotting the trade-off between **True Positive Rate (TPR)** and **False Positive Rate (FPR)** across different thresholds.

- **True Positive Rate (TPR)** (also known as **Recall**):
  
  \[
  \text{TPR} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  \]

- **False Positive Rate (FPR)**:
  
  \[
  \text{FPR} = \frac{\text{False Positives (FP)}}{\text{False Positives (FP)} + \text{True Negatives (TN)}}
  \]

The ROC curve is generated by calculating TPR and FPR at various thresholds. A model that performs well will have a curve that hugs the upper left corner.

### Example ROC Curve

Here’s a Python example to generate an ROC curve:

```python
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Simulated true labels and model predicted probabilities
y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.5, 0.3, 0.9, 0.65, 0.2])

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label='ROC Curve')

# Labeling the axes
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# Show the plot
plt.legend()
plt.show()
```

### AUC (Area Under the Curve)

- The **AUC** quantifies the overall performance of the model by measuring the area under the ROC curve.
  
  - **AUC = 1**: Perfect model.
  - **AUC = 0.5**: No better than random guessing.

### Applications

- **Model Comparison**: AUC helps compare different models.
- **Threshold Selection**: The ROC curve aids in selecting the optimal threshold based on the trade-off between TPR and FPR.

---

The ROC curve can only be plotted for models that output predicted probabilities, such as logistic regression and random forest. Other models such as Naive Bayes do not, and therefore ROC curves cannot be plotted for these models.
In order to compare the performance of different classifiers, the standard practice is to plot the ROC curve and then compare the AUC values. The model with the higher AUC value is considered to be more performant.

## Partial ROC Curve

### Introduction

The **Partial ROC Curve (pROC)** focuses on a specific portion of the ROC curve, typically the range where **False Positive Rate (FPR)** is low. This is especially useful in domains where minimizing false positives is crucial, such as **medical diagnosis** or **fraud detection**.

### Example Partial ROC Curve

Here’s a Python example to generate a Partial ROC Curve:

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Simulated true labels and model predicted probabilities
y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.5, 0.3, 0.9, 0.65, 0.2])

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Define range of FPR for the partial ROC curve (e.g., FPR < 0.1)
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
plt.show()
```

### Applications of Partial ROC

- **Medical Diagnosis**: Avoiding false positives is critical, so performance at low FPR is emphasized.
- **Fraud Detection**: Limiting false positives prevents unnecessary investigations.
- **Anomaly Detection**: Ensures minimal false alarms by focusing on low FPR ranges.

### Partial AUC (pAUC)

The **Partial AUC (pAUC)** measures the area under the ROC curve within a specified FPR range, providing a more relevant metric in scenarios where low FPR is crucial.

### Advantages of pROC

- **Focus on Relevant Performance**: Highlights performance in regions of interest, such as low FPR.
- **Better Decision-Making**: Allows stakeholders to make informed choices when high false positives are unacceptable.

### Limitations of pROC

- **Arbitrary FPR Threshold**: The selection of an FPR threshold can be subjective and domain-dependent.
- **Loss of Full Picture**: Focusing on a partial region may hide useful information about model performance at higher FPR values.

---

### Summary

- **Precision-Recall Curve** is ideal for imbalanced datasets, showing the trade-off between precision and recall.
- **ROC Curve** provides a comprehensive view of a model’s classification performance.
- **Partial ROC Curve** zooms in on regions of interest (low FPR), making it crucial in domains where controlling false positives is important.

# Primary Scoring Metric
Submissions are evaluated on partial area under the ROC curve (pAUC) above 80% true positive rate (TPR) for binary classification of malignant examples. (See the implementation in the notebook ISIC pAUC-aboveTPR.)

The receiver operating characteristic (ROC) curve illustrates the diagnostic ability of a given binary classifier system as its discrimination threshold is varied. However, there are regions in the ROC space where the values of TPR are unacceptable in clinical practice. Systems that aid in diagnosing cancers are required to be highly-sensitive, so this metric focuses on the area under the ROC curve AND above 80% TRP. Hence, scores range from [0.0, 0.2].

```python
"""
2024 ISIC Challenge primary prize scoring metric

Given a list of binary labels, an associated list of prediction 
scores ranging from [0,1], this function produces, as a single value, 
the partial area under the receiver operating characteristic (pAUC) 
above a given true positive rate (TPR).
https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

(c) 2024 Nicholas R Kurtansky, MSKCC
"""

import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import roc_curve, auc, roc_auc_score

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80) -> float:
    '''
    2024 ISIC Challenge metric: pAUC
    
    Given a solution file and submission file, this function returns the
    the partial area under the receiver operating characteristic (pAUC) 
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.
    
    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        solution: ground truth pd.DataFrame of 1s and 0s
        submission: solution dataframe of predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # check submission is numeric
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('Submission target column must be numeric')

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution.values)-1)
    
    # flip the submissions to their compliments
    v_pred = -1.0*np.asarray(submission.values)

    max_fpr = abs(1-min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

#     # Equivalent code that uses sklearn's roc_auc_score
#     v_gt = abs(np.asarray(solution.values)-1)
#     v_pred = np.array([1.0 - x for x in submission.values])
#     max_fpr = abs(1-min_tpr)
#     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
#     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
#     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
#     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return(partial_auc)


```







