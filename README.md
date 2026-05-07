# Instacart Analytics Dashboard

A Streamlit-powered analytics dashboard for retail (grocery store/Instacart-style) order analysis, product reorder predictions, customer segmentation, and market basket (association rule) analysis.  
This project demonstrates full ML workflow: data cleaning, feature engineering, model training (Random Forest & Logistic Regression), clustering, and deploying results to an interactive dashboard.

---

## 📊 Features

- **Data Insights:** Visualize user order patterns, product popularity, and reorder trends.
- **Product Reorder Prediction:** Predict the likelihood of a user reordering a specific product using ML models.
- **Customer Segmentation:** Segment customers into clusters (e.g., loyal, occasional, high-reorder, and low-engagement) using KMeans.
- **Market Basket Analysis:** Discover product pairs frequently bought together using association rules.
- **Interactive Dashboard:** All insights accessible through a modern multipage Streamlit UI.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ThiwankaKT/ML_Project.git
    cd ML_Project
    ```

2. **(Optional) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate         # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Usage

1. **Prepare Data and Models**
    - Run `shopping_analysis.ipynb` in Jupyter or VS Code to:
        - Clean and preprocess the raw datasets.
        - Generate CSVs (`*_sample.csv`) required by the dashboard.
        - Train and export models (Random Forest, Logistic Regression) and scaler as `.pkl` files.
        - Output association rules for market basket analysis.
    - Skip this step if sample datasets & models already exist.

2. **Launch the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    The dashboard will open in your browser.

3. **Interact with the Dashboard**
    - Explore data insights (tabs for visual analytics).
    - Predict reorder likelihood by choosing a user and product.
    - See customer clusters with summary metrics.
    - View top product pairs bought together.

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit dashboard
├── shopping_analysis.ipynb # Data analysis, ML training, EDA, and feature creation
├── requirements.txt
├── rf_model.pkl            # Trained Random Forest model
├── log_model.pkl           # Trained Logistic Regression model
├── scaler.pkl              # Fitted feature scaler
├── *_sample.csv            # Sample datasets for app inference
├── basket_analysis.csv     # Association rules (frequently bought product pairs)
├── cluster_summary.csv     # Customer segment summary
├── user_clusters.csv       # User cluster assignments
└── ... (other datasets)
```

---

## 🧠 Models & Methods

- **Classification:**  
  Logistic Regression & Random Forest to predict product reorder.
- **Segmentation:**  
  KMeans clustering for user segments based on order history and reorder behavior.
- **Association Rules:**  
  MLxtend's Apriori and association rules to find product bundles.

---

## 📦 Requirements

```txt
streamlit>=1.32
pandas
numpy
scikit-learn
joblib
plotly
networkx
matplotlib
seaborn
```

---

## 📚 Notebooks & Data

- Main EDA, feature engineering, and training are done in `shopping_analysis.ipynb`.
- **Input Files:** See CSVs in the repo (e.g., `orders_sample.csv`, `order_details_sample.csv`, etc.)
- **Outputs:** Preprocessed datasets and trained models, ready for use with `app.py`.

---

## 🤝 Contributing

Pull Requests welcome! If you wish to add more analysis modules, models, or visualizations, please open an issue to discuss your idea.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Inspired by retail reorder datasets (e.g., Instacart Market Basket Analysis).
- Built with Streamlit, scikit-learn, pandas, Plotly, and MLxtend.
