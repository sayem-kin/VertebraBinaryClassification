### 🔍 Results (First Commit)

This first commit contains a **full end-to-end pipeline** for a **vanilla binary classifier** using SE-ResNet.  
The goal was to establish a baseline and store the raw results for comparison with future improvements.

**Summary:**  
The model performs poorly on the minority class, showing strong class imbalance issues and limited generalization.

**Key Metrics**
- **Test Accuracy:** 0.7650  
- **Test F1:** 0.2321  
- **Test AUC:** 0.7244  

**Classification Report**
- Class **0** — Precision: 0.8190, Recall: 0.9082, F1: 0.8613 (Support: 294)  
- Class **1** — Precision: 0.3250, Recall: 0.1806, F1: 0.2321 (Support: 72)  

**Confusion Matrix**

[[267  27]  
 [ 59  13]]

> The model heavily favors the majority class (0). Later experiments will focus on handling class imbalance and improving minority-class performance.
# Improvement

## Key Improvements (Compared to Previous Version)

- **WeightedRandomSampler** to balance the 4:1 class imbalance.
- **Focal Loss** to improve fracture detection (minority class).
- **Improved HU windowing** (`WL=200, WW=700`) for better vertebral body contrast.
- **Geometric data augmentation** (rotation + affine transforms) to reduce overfitting.
- **Optional EfficientNet-B3 backbone** for stronger feature extraction.
- **CosineAnnealingLR** for smoother learning rate scheduling.
- Updated dataset pipeline with standardized preprocessing and augmentation.

## Updated Performance (Test Set)

| Metric | Value |
|--------|--------|
| Accuracy | **0.8443** |
| AUC | **0.7721** |
| F1 (Fracture) | **0.5778** |
| Recall (Fracture) | **0.5417** |
| Precision (Fracture) | **0.6190** |

These results show a **major improvement**, particularly in fracture detection sensitivity and F1 score, making this classifier suitable for downstream tasks like Grad-CAM visualization.

## 📁 Project Structure

