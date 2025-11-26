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
