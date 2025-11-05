# YouTube CDN Traffic Analysis

This repository contains a **Machine Learning model** and the corresponding **project report** analyzing YouTube’s Content Delivery Network (CDN) traffic.

*Authors: Antonello Di Pede, Alex Cugliari, Giuseppe Insalaco*
Politecnico Di Torino
---

## 🧠 Overview
The project investigates YouTube CDN behavior using real network data.  
It applies **data pre-processing, regression, and clustering** techniques to estimate **flow throughput** and detect **topological changes** in the CDN structure.

---

## ⚙️ Methods
- **Data filtering and outlier removal**
- **PCA** for dimensionality reduction  
- **Supervised learning:** Linear Regression, Lasso, KNN  
- **Unsupervised learning:** K-Means clustering to group edge-nodes  
- **Evaluation:** MSE, R², Silhouette, Davies-Bouldin, Calinski-Harabasz indices

---

## 📂 Repository Structure
youtube-cdn-ml-analysis/
│
├── model.py       # Python script for ML and clustering analysis
├── report.pdf     # Final written report (full documentation)
└── README.md      # Project overview


---

## 📈 Key Insights
- Linear and Lasso regression achieved **R² ≈ 0.99**  
- K-Means detected **5 main CDN clusters**  
- Cluster evolution analysis revealed **structural changes after Week 3**

---
