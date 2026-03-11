# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page config and custom styling
# -------------------------
st.set_page_config(
    page_title="Instacart Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background: linear-gradient(to right, #f8f9fa, #e3f2fd);}
[data-testid="stSidebar"] {background-color: #f4f6f8; padding: 20px; border-radius: 10px;}
h1,h2,h3 {font-family: 'Arial', sans-serif;}
.stDataFrame th {text-align:center !important;}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Instacart Analytics Dashboard")
st.markdown("Explore product reorder predictions, customer insights, and frequently bought together products.")

# -------------------------
# Load precomputed models and scaler
# -------------------------
rf_model = joblib.load("rf_model.pkl")
log_model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# Load pre-saved CSVs from notebook
# -------------------------
orders = pd.read_csv("orders_sample.csv")
order_details = pd.read_csv("order_details_sample.csv")
products = pd.read_csv("products_lookup.csv")
aisles = pd.read_csv("aisles_sample.csv")
departments = pd.read_csv("departments_sample.csv")
ml_features = pd.read_csv("ml_features.csv")
user_clusters = pd.read_csv("user_clusters.csv")

# Load basket analysis
try:
    basket_rules = pd.read_csv("basket_analysis.csv")
except FileNotFoundError:
    basket_rules = pd.DataFrame()

# Pre-merge for prediction
order_products_prior_user = order_details[['user_id','product_id','order_number','days_since_prior_order','reordered']]

# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("## ⚙️ Model Settings")
model_choice = st.sidebar.radio("Choose Prediction Model", ["Random Forest", "Logistic Regression"])
st.sidebar.markdown("---")
confidence_placeholder = st.sidebar.empty()

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Insights 📊",
    "Product Reorder Prediction 🔄",
    "Customer Insights & Segments 👥",
    "Frequently Bought Together 🛒"
])

# -------------------------
# Tab 0: Insights
# -------------------------
with tabs[0]:
    st.header("Exploratory Data Insights 📊")
    col1, col2 = st.columns(2)

    # Orders per day
    orders_per_day = orders.groupby("order_dow")["order_id"].count().reset_index()
    orders_per_day.rename(columns={"order_id":"Num Orders"}, inplace=True)
    fig1 = px.bar(orders_per_day, x="order_dow", y="Num Orders",
                  color="Num Orders", color_continuous_scale="Blues",
                  title="Orders per Day of Week")
    col1.plotly_chart(fig1, use_container_width=True)

    # Orders per hour
    orders_per_hour = orders.groupby("order_hour_of_day")["order_id"].count().reset_index()
    orders_per_hour.rename(columns={"order_id":"Num Orders"}, inplace=True)
    fig2 = px.bar(orders_per_hour, x="order_hour_of_day", y="Num Orders",
                  color="Num Orders", color_continuous_scale="Oranges",
                  title="Orders per Hour of Day")
    col2.plotly_chart(fig2, use_container_width=True)

    # Top products
    col3, col4 = st.columns(2)
    top_products = order_details.groupby("product_name")["order_id"].count().reset_index().sort_values("order_id", ascending=False).head(10)
    fig3 = px.bar(top_products, x="order_id", y="product_name", orientation="h",
                  color="order_id", color_continuous_scale="Viridis",
                  title="Top 10 Products by Orders", labels={"order_id":"Num Orders"})
    fig3.update_layout(yaxis={'categoryorder':'total ascending'})
    col3.plotly_chart(fig3, use_container_width=True)

    # Reorder rate by department
    dept_reorder = order_details.groupby("department")["reordered"].mean().reset_index()
    fig4 = px.bar(dept_reorder, x="department", y="reordered",
                  color="reordered", color_continuous_scale="Teal",
                  title="Average Reorder Rate by Department",
                  labels={"reordered":"Avg Reorder Rate"})
    col4.plotly_chart(fig4, use_container_width=True)

# -------------------------
# Tab 1: Product Reorder Prediction
# -------------------------
with tabs[1]:
    st.subheader("🔄 Product Reorder Prediction")
    col1, col2 = st.columns([1,1])
    
    with col1:
        user_id = st.selectbox("Select User", sorted(orders["user_id"].unique()))
        product_name = st.selectbox("Select Product", sorted(products["product_name"].unique()))
        predict_button = st.button("Predict")

    with col2:
        if predict_button:
            product_id = products.loc[products["product_name"]==product_name,"product_id"].values[0]
            user_data = orders[orders["user_id"]==user_id]
            total_orders = user_data["order_number"].max()
            avg_days_between_orders = user_data["days_since_prior_order"].mean()
            prod_data = order_details[order_details["product_id"]==product_id]
            product_popularity = len(prod_data)
            product_reorder_rate = prod_data["reordered"].mean()
            user_prod = order_products_prior_user[
                (order_products_prior_user["user_id"]==user_id) &
                (order_products_prior_user["product_id"]==product_id)
            ]
            purchase_count = len(user_prod)
            features = np.array([[total_orders, avg_days_between_orders,
                                  product_popularity, product_reorder_rate, purchase_count]])
            
            # Model prediction
            if model_choice=="Logistic Regression":
                features_scaled = scaler.transform(features)
                prediction = log_model.predict(features_scaled)[0]
                pred_prob = log_model.predict_proba(features_scaled)[0][1]
            else:
                prediction = rf_model.predict(features)[0]
                pred_prob = rf_model.predict_proba(features)[0][1]

            st.subheader("Prediction Result")
            if prediction==1:
                st.success("Likely to be reordered")
            else:
                st.error("Unlikely to be reordered")
            st.metric("Reorder Probability", f"{pred_prob:.2%}")

# -------------------------
# Tab 2: Customer Insights & Segments
# -------------------------
with tabs[2]:
    st.header("Customer Insights & Segments")
    cluster_summary = user_clusters.groupby('cluster').agg(
        total_users=('user_id','count'),
        avg_orders=('total_orders','mean'),
        avg_days_between=('avg_days_between_orders','mean')
    ).reset_index()
    st.subheader("Cluster Overview")
    cols = st.columns(len(cluster_summary))
    cluster_colors = {0:"#4CAF50",1:"#2196F3",2:"#FF9800",3:"#9E9E9E"}
    for i,row in cluster_summary.iterrows():
        with cols[i]:
            st.markdown(f"""
            <div style="background-color:{cluster_colors[row['cluster']]};padding:15px;border-radius:10px;color:white;text-align:center;">
            <h3>Cluster {row['cluster']}</h3>
            <p><strong>Total Users:</strong> {row['total_users']}</p>
            <p><strong>Avg Orders:</strong> {row['avg_orders']:.1f}</p>
            <p><strong>Avg Days Between Orders:</strong> {row['avg_days_between']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------
# Tab 3: Frequently Bought Together
# -------------------------
with tabs[3]:
    st.header("Frequently Bought Together 🛒")
    if not basket_rules.empty:
        basket_rules['antecedents'] = basket_rules['antecedents'].apply(lambda x: ', '.join(eval(x)) if isinstance(x,str) else ', '.join(x))
        basket_rules['consequents'] = basket_rules['consequents'].apply(lambda x: ', '.join(eval(x)) if isinstance(x,str) else ', '.join(x))
        basket_rules['pair_sorted'] = basket_rules.apply(lambda r: ' 🛒 '.join(sorted([r['antecedents'],r['consequents']])), axis=1)
        top_rules = basket_rules.drop_duplicates('pair_sorted').sort_values('support', ascending=False).head(10)
        st.dataframe(top_rules[['pair_sorted','support']].rename(columns={'pair_sorted':'Products Bought Together','support':'Support'}))
    else:
        st.info("ℹ️ No product pairs to display. Run the notebook first to generate basket_analysis.csv.")