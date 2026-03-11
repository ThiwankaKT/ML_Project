# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# Page config and custom styling
# --------------------------------
st.set_page_config(
    page_title="Instacart Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background and cards
st.markdown("""
<style>
/* Background gradient */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(to right, #f8f9fa, #e3f2fd);
}

/* Sidebar style */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    padding: 20px;
}

/* Card text alignment */
.stDataFrame th {
    text-align: center !important;
}

h1, h2, h3 {
    font-family: 'Arial', sans-serif;
}

/* Sidebar background and padding */            
[data-testid="stSidebar"] {
    background-color: #f4f6f8;  /* light gray */
    padding: 20px;
    border-radius: 10px;
}

/* Sidebar headers styling */
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    font-family: 'Arial', sans-serif;
    color: #2c3e50;  /* dark blue-gray */
}

/* Sidebar radio buttons spacing */
[data-testid="stSidebar"] .stRadio {
    margin-bottom: 15px;
}

/* Sidebar info box */
[data-testid="stSidebar"] .stInfo {
    font-size: 0.9rem;
    background-color: #e8f0fe;
    border-left: 4px solid #4285f4;
    padding: 10px;
    border-radius: 5px;
}
            
</style>
""", unsafe_allow_html=True)

# --------------------------------
# App title
# --------------------------------
st.title("🛒 Instacart Analytics Dashboard")
st.markdown("Explore product reorder predictions, customer insights, and frequently bought together products.")

# -----------------------------
# Load models and scaler
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
log_model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Load datasets
# -----------------------------
orders = pd.read_csv("datasets/orders.csv")
order_products_prior = pd.read_csv("datasets/order_products_prior.csv")
products = pd.read_csv("datasets/products.csv")
aisles = pd.read_csv("datasets/aisles.csv")
departments = pd.read_csv("datasets/departments.csv")

# Merge user_id into prior orders
order_products_prior_user = order_products_prior.merge(
    orders[['order_id', 'user_id', 'order_number', 'days_since_prior_order']],
    on='order_id', how='left'
)

# Load basket analysis results
try:
    basket_rules = pd.read_csv("basket_analysis.csv")
except FileNotFoundError:
    basket_rules = pd.DataFrame()

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.markdown("## ⚙️ Model Settings", unsafe_allow_html=True)

model_choice = st.sidebar.radio(
    "Choose Prediction Model",
    ["Random Forest", "Logistic Regression"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Model Performance")

confidence_placeholder = st.sidebar.empty()

st.markdown("""
<style>

.main-title{
    font-size:32px;
    font-weight:700;
}

.input-card{
    background-color:#f7f9fc;
    padding:20px;
    border-radius:12px;
    border:1px solid #e6e6e6;
}

.result-card{
    background-color:#ffffff;
    padding:25px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    box-shadow:0 4px 10px rgba(0,0,0,0.05);
}

/* Prevent empty containers from reserving space */
.input-card, .result-card {
    min-height: 0 !important;
    display: inline-block !important;
    overflow: hidden !important;
    margin: 0 !important;
}

.metric-box{
    background-color:#f1f5f9;
    padding:15px;
    border-radius:10px;
    margin-top:10px;
}

.feature-box{
    background-color:#f8fafc;
    padding:20px;
    border-radius:12px;
    border:1px solid #e6e6e6;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs for sections
# -----------------------------
tabs = st.tabs([
    "Product Reorder Prediction 🔄",
    "Customer Insights & Segments 👥",
    "Frequently Bought Together 🛒"
])

# -----------------------------
# Tab 1: Product Reorder Prediction
# -----------------------------
with tabs[0]:

    # Title and description
    st.markdown('<div class="main-title">🔄 Product Reorder Prediction</div>', unsafe_allow_html=True)
    st.write("Select a user and product to predict whether the product will likely be reordered.")

    # Create two columns: left = inputs, right = results
    col1, col2 = st.columns([1, 1])  # equal width

    # -------------------------
    # INPUTS
    # -------------------------
    with col1:
        st.subheader("Select Inputs")
        user_id = st.selectbox("User", sorted(orders["user_id"].unique()))
        product_name = st.selectbox("Product", sorted(products["product_name"].unique()))
        predict_button = st.button("Predict")

    # -------------------------
    # RESULTS (only after clicking)
    # -------------------------
    with col2:
        if predict_button:
            st.subheader("Prediction Result")

            # Extract features
            product_id = products.loc[products["product_name"] == product_name, "product_id"].values[0]
            user_orders = orders[orders["user_id"] == user_id]
            total_orders = user_orders["order_number"].max()
            avg_days_between_orders = user_orders["days_since_prior_order"].mean()

            product_orders = order_products_prior[order_products_prior["product_id"] == product_id]
            product_popularity = len(product_orders)
            product_reorder_rate = product_orders["reordered"].mean()

            user_product = order_products_prior_user[
                (order_products_prior_user["user_id"] == user_id) &
                (order_products_prior_user["product_id"] == product_id)
            ]
            purchase_count = len(user_product)

            features = np.array([[total_orders, avg_days_between_orders,
                                  product_popularity, product_reorder_rate, purchase_count]])

            # Model prediction
            if model_choice == "Logistic Regression":
                features_scaled = scaler.transform(features)
                prediction = log_model.predict(features_scaled)[0]
                pred_prob = log_model.predict_proba(features_scaled)[0][1]
            else:
                prediction = rf_model.predict(features)[0]
                pred_prob = rf_model.predict_proba(features)[0][1]

            # Show result
            if prediction == 1:
                st.success("Likely to be reordered")
            else:
                st.error("Unlikely to be reordered")

            st.metric("Reorder Probability", f"{pred_prob:.2%}")

            # Determine color based on probability
            if pred_prob >= 0.75:
                bar_color = "#4CAF50"  # green
            elif pred_prob >= 0.5:
                bar_color = "#FFC107"  # yellow
            else:
                bar_color = "#F44336"  # red

            # Display progress bar with HTML/CSS
            confidence_html = f"""
            <div style="
                padding:5px;
                border-radius:12px;
                background-color:#e0e0e0;
                width:100%;
                margin-bottom:10px;
            ">
                <div style="
                    width:{pred_prob*100}%;
                    background-color:{bar_color};
                    height:30px;
                    border-radius:12px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-weight:bold;
                    color:white;
                    font-size:14px;
                ">
                    {pred_prob:.0%} Confidence
                </div>
            </div>
            """

            confidence_placeholder.markdown(confidence_html, unsafe_allow_html=True)

            # Feature display
            st.subheader("Calculated Features")
            feature_display = pd.DataFrame({
                "Feature": ["Total Orders", "Average Days Between Orders",
                            "Product Popularity", "Product Reorder Rate", "User Purchase Count"],
                "Value": [total_orders, round(avg_days_between_orders, 2),
                          product_popularity, round(product_reorder_rate, 2), purchase_count]
            })
            st.dataframe(feature_display, use_container_width=True)

# -----------------------------
# Tab 2: Customer Insights & Segments
# -----------------------------
with tabs[1]:
    st.header("Customer Insights & Segments")
    st.write("Explore the main customer groups and their behavior at a glance.")

    customer_features = order_products_prior_user.groupby('user_id').agg(
        total_orders=('order_number','max'),
        avg_days=('days_since_prior_order','mean'),
        reorder_rate=('reordered','mean')
    ).fillna(0).reset_index()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_features['cluster'] = kmeans.fit_predict(
        customer_features[['total_orders','avg_days','reorder_rate']]
    )

    cluster_map = {
        0: "🎯 Loyal Frequent Shoppers",
        1: "🛍️ Occasional Shoppers",
        2: "🔄 High Reorder Customers",
        3: "😴 Low Engagement Customers"
    }
    emoji_map = {v:v.split(' ')[0] for v in cluster_map.values()}
    customer_features['cluster_name'] = customer_features['cluster'].map(cluster_map)

    cluster_summary = customer_features.groupby('cluster_name').agg(
        total_users=('user_id','count'),
        avg_orders=('total_orders','mean'),
        avg_days_between=('avg_days','mean'),
        avg_reorder_rate=('reorder_rate','mean')
    ).reset_index()

    # Cards
    st.subheader("Cluster Overview")
    cols = st.columns(len(cluster_summary))
    cluster_colors = {
        "🎯 Loyal Frequent Shoppers":"#4CAF50",
        "🛍️ Occasional Shoppers":"#2196F3",
        "🔄 High Reorder Customers":"#FF9800",
        "😴 Low Engagement Customers":"#9E9E9E"
    }
    for i, row in cluster_summary.iterrows():
        with cols[i]:
            st.markdown(f"""
            <div style="
                background-color:{cluster_colors[row['cluster_name']]};
                padding:15px;
                border-radius:10px;
                color:white;
                text-align:center;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                margin-bottom:10px;
                height:300px;
                display:flex;
                flex-direction:column;
                justify-content:center;">
                <h3>{emoji_map[row['cluster_name']]} {row['cluster_name']}</h3>
                <p><strong>Total Users:</strong> {row['total_users']:,}</p>
                <p><strong>Avg Orders:</strong> {row['avg_orders']:.1f}</p>
                <p><strong>Avg Days Between Orders:</strong> {row['avg_days_between']:.1f}</p>
                <p><strong>Avg Reorder Rate:</strong> {row['avg_reorder_rate']:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

    # Bubble chart
    fig = px.scatter(
        cluster_summary,
        x='avg_orders',
        y='avg_reorder_rate',
        size='total_users',
        color='avg_reorder_rate',
        hover_name='cluster_name',
        size_max=80,
        color_continuous_scale='Viridis',
        title="Customer Segments Overview: Orders vs Reorder Rate"
    )
    fig.update_layout(xaxis_title="Average Orders per User", yaxis_title="Average Reorder Rate")
    st.plotly_chart(fig, use_container_width=True)

    # Drill-down table
    st.subheader("Explore Users in a Cluster")
    selected_cluster = st.selectbox("Select Cluster", cluster_summary['cluster_name'].tolist())
    top_users = customer_features[customer_features['cluster_name']==selected_cluster].sort_values(
        'reorder_rate', ascending=False
    ).head(10)[['user_id','total_orders','avg_days','reorder_rate']]
    top_users = top_users.rename(columns={
        'user_id':'User ID','total_orders':'Total Orders',
        'avg_days':'Avg Days Between Orders','reorder_rate':'Reorder Rate'
    })
    top_users['Reorder Rate'] = (top_users['Reorder Rate']*100).round(1).astype(str) + '%'
    st.dataframe(top_users.reset_index(drop=True))

# -----------------------------
# Tab 3: Frequently Bought Together
# -----------------------------
with tabs[2]:
    st.header("Frequently Bought Together 🛒")
    st.write("See which products are commonly purchased together. Great for recommendations or promotions!")

    if not basket_rules.empty:
        # Convert frozensets to readable strings
        basket_rules['antecedents'] = basket_rules['antecedents'].apply(
            lambda x: ', '.join(eval(x)) if isinstance(x,str) else ', '.join(x)
        )
        basket_rules['consequents'] = basket_rules['consequents'].apply(
            lambda x: ', '.join(eval(x)) if isinstance(x,str) else ', '.join(x)
        )

        # Remove duplicates like A->B and B->A
        basket_rules['pair_sorted'] = basket_rules.apply(
            lambda row: ' 🛒 '.join(sorted([row['antecedents'], row['consequents']])), axis=1
        )
        basket_rules_unique = basket_rules.drop_duplicates(subset='pair_sorted')
        basket_rules_unique['times_bought_together'] = (basket_rules_unique['support']*10000).astype(int)

        # Top 10 pairs
        top_rules = basket_rules_unique.sort_values('times_bought_together', ascending=False).head(10)
        top_rules = top_rules[['pair_sorted','times_bought_together']].rename(
            columns={'pair_sorted':'Products Bought Together','times_bought_together':'Times Bought Together'}
        ).reset_index(drop=True)

        # Display cards
        st.subheader("Top Product Pairs")
        cols = st.columns(2)  # two cards per row
        for i, row in top_rules.iterrows():
            col = cols[i % 2]
            # intensity for color: darker for higher frequency
            max_val = top_rules['Times Bought Together'].max()
            intensity = int(255 - (row['Times Bought Together']/max_val)*100)
            bg_color = f'rgb(135, {intensity}, 200)'

            with col:
                st.markdown(f"""
                <div style="
                    background-color:{bg_color};
                    padding:20px;
                    border-radius:12px;
                    color:white;
                    text-align:center;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
                    margin-bottom:15px;">
                    <h4 style="margin:5px;">{row['Products Bought Together']}</h4>
                    <p style="margin:5px;"><strong>Times Bought Together:</strong> {row['Times Bought Together']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        **💡 Recommendations:**  
        - Bundle these products in promotions.  
        - Recommend to users buying one item in the pair.  
        - Adjust inventory and marketing based on popular combinations.
        """)

    else:
        st.info("ℹ️ No product pairs to display. Run the notebook first to generate basket_analysis.csv.")