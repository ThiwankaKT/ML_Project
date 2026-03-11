# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

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
# Load CSV datasets from notebook
# -----------------------------
orders = pd.read_csv("orders_sample.csv")
order_details = pd.read_csv("order_details_sample.csv")
products = pd.read_csv("products_lookup.csv")
aisles = pd.read_csv("aisles_sample.csv")
departments = pd.read_csv("departments_sample.csv")

# Merge user_id into prior orders for predictions
order_products_prior_user = order_details[['order_id','user_id','order_number','days_since_prior_order','product_id','reordered']]

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
confidence_placeholder = st.sidebar.empty()

# -----------------------------
# Tabs for sections
# -----------------------------
tabs = st.tabs([
    "Insights 📊",
    "Product Reorder Prediction 🔄",
    "Customer Insights & Segments 👥",
    "Frequently Bought Together 🛒"
])

# -----------------------------
# Tab 0: Insights
# -----------------------------
with tabs[0]:
    st.header("Exploratory Data Insights 📊")
    st.markdown("Quick visual overview of user behavior, product popularity, and reorder trends.")

    col1, col2 = st.columns(2)

    # Orders per Day of Week
    orders_per_day = orders.groupby("order_dow")["order_id"].count().reset_index()
    orders_per_day.rename(columns={"order_id":"Num Orders"}, inplace=True)
    fig1 = px.bar(
        orders_per_day,
        x="order_dow",
        y="Num Orders",
        labels={"order_dow":"Day of Week"},
        color="Num Orders",
        color_continuous_scale="Blues",
        title="Orders per Day of Week"
    )
    fig1.update_layout(showlegend=False)
    col1.plotly_chart(fig1, use_container_width=True)

    # Orders per Hour
    orders_per_hour = orders.groupby("order_hour_of_day")["order_id"].count().reset_index()
    orders_per_hour.rename(columns={"order_id":"Num Orders"}, inplace=True)
    fig2 = px.bar(
        orders_per_hour,
        x="order_hour_of_day",
        y="Num Orders",
        labels={"order_hour_of_day":"Hour of Day"},
        color="Num Orders",
        color_continuous_scale="Oranges",
        title="Orders per Hour of Day"
    )
    fig2.update_layout(showlegend=False)
    col2.plotly_chart(fig2, use_container_width=True)

    # Second row: Top 10 Products by Popularity
    col3, col4 = st.columns(2)

    top_products = order_details.groupby("product_id")["order_id"].count().reset_index()
    top_products = top_products.merge(products[["product_id","product_name"]], on="product_id")
    top_products = top_products.sort_values("order_id", ascending=False).head(10)
    fig3 = px.bar(
        top_products,
        x="order_id",
        y="product_name",
        orientation="h",
        color="order_id",
        color_continuous_scale="Viridis",
        title="Top 10 Products by Orders",
        labels={"order_id":"Num Orders","product_name":"Product"}
    )
    fig3.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    col3.plotly_chart(fig3, use_container_width=True)

    # Reorder Rate by Department
    dept_reorder = order_details.groupby("department_id")["reordered"].mean().reset_index()
    dept_reorder = dept_reorder.merge(departments, on="department_id")
    fig4 = px.bar(
        dept_reorder,
        x="department",
        y="reordered",
        color="reordered",
        color_continuous_scale="Teal",
        title="Average Reorder Rate by Department",
        labels={"reordered":"Avg Reorder Rate","department":"Department"}
    )
    col4.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Tab 1: Product Reorder Prediction
# -----------------------------
with tabs[1]:
    st.markdown("## 🔄 Product Reorder Prediction")
    st.write("Select a user and product to predict whether the product will likely be reordered.")

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.selectbox("User", sorted(orders["user_id"].unique()))
        product_name = st.selectbox("Product", sorted(products["product_name"].unique()))
        predict_button = st.button("Predict")

    with col2:
        if predict_button:
            # Feature extraction
            product_id = products.loc[products["product_name"]==product_name,"product_id"].values[0]
            user_orders = orders[orders["user_id"]==user_id]
            total_orders = user_orders["order_number"].max()
            avg_days_between_orders = user_orders["days_since_prior_order"].mean()

            product_orders = order_details[order_details["product_id"]==product_id]
            product_popularity = len(product_orders)
            product_reorder_rate = product_orders["reordered"].mean()

            user_product = order_products_prior_user[
                (order_products_prior_user["user_id"]==user_id) &
                (order_products_prior_user["product_id"]==product_id)
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

            # Display
            if prediction==1:
                st.success("Likely to be reordered")
            else:
                st.error("Unlikely to be reordered")

            st.metric("Reorder Probability", f"{pred_prob:.2%}")

            # Display confidence bar in sidebar
            if pred_prob >= 0.75:
                bar_color = "#4CAF50"  # green
            elif pred_prob >= 0.5:
                bar_color = "#FFC107"  # yellow
            else:
                bar_color = "#F44336"  # red

            confidence_html = f"""
            <div style="
                padding:5px;
                border-radius:12px;
                background-color:#e0e0e0;
                width:100%;
                margin-bottom:10px;">
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
                    font-size:14px;">
                    {pred_prob:.0%} Confidence
                </div>
            </div>
            """

            # Update the sidebar placeholder
            confidence_placeholder.markdown(confidence_html, unsafe_allow_html=True)

            # Feature table
            feature_display = pd.DataFrame({
                "Feature":["Total Orders","Average Days Between Orders","Product Popularity","Product Reorder Rate","User Purchase Count"],
                "Value":[total_orders, round(avg_days_between_orders,2), product_popularity, round(product_reorder_rate,2), purchase_count]
            })
            st.dataframe(feature_display,use_container_width=True)

# -----------------------------
# Tab 2: Customer Insights & Segments
# -----------------------------
with tabs[2]:
    st.header("Customer Insights & Segments")

    customer_features = order_products_prior_user.groupby('user_id').agg(
        total_orders=('order_number','max'),
        avg_days=('days_since_prior_order','mean'),
        reorder_rate=('reordered','mean')
    ).fillna(0).reset_index()

    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_features['cluster'] = kmeans.fit_predict(customer_features[['total_orders','avg_days','reorder_rate']])

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

    # Cluster cards
    st.subheader("Cluster Overview")
    cols = st.columns(len(cluster_summary))
    cluster_colors = {
        "🎯 Loyal Frequent Shoppers":"#4CAF50",
        "🛍️ Occasional Shoppers":"#2196F3",
        "🔄 High Reorder Customers":"#FF9800",
        "😴 Low Engagement Customers":"#9E9E9E"
    }
    for i,row in cluster_summary.iterrows():
        with cols[i]:
            st.markdown(f"""
            <div style="background-color:{cluster_colors[row['cluster_name']]};
                        padding:15px; border-radius:10px; color:white; text-align:center;
                        box-shadow:2px 2px 5px rgba(0,0,0,0.3); margin-bottom:10px;
                        height:300px; display:flex; flex-direction:column; justify-content:center;">
                <h3>{emoji_map[row['cluster_name']]} {row['cluster_name']}</h3>
                <p><strong>Total Users:</strong> {row['total_users']:,}</p>
                <p><strong>Avg Orders:</strong> {row['avg_orders']:.1f}</p>
                <p><strong>Avg Days Between Orders:</strong> {row['avg_days_between']:.1f}</p>
                <p><strong>Avg Reorder Rate:</strong> {row['avg_reorder_rate']:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

    # Bubble chart
    fig = px.scatter(cluster_summary, x='avg_orders', y='avg_reorder_rate',
                     size='total_users', color='avg_reorder_rate',
                     hover_name='cluster_name', size_max=80, color_continuous_scale='Viridis',
                     title="Customer Segments Overview: Orders vs Reorder Rate")
    fig.update_layout(xaxis_title="Average Orders per User", yaxis_title="Average Reorder Rate")
    st.plotly_chart(fig,use_container_width=True)

    # Drill-down table
    st.subheader("Explore Users in a Cluster")
    selected_cluster = st.selectbox("Select Cluster", cluster_summary['cluster_name'].tolist())
    top_users = customer_features[customer_features['cluster_name']==selected_cluster].sort_values('reorder_rate',ascending=False).head(10)
    top_users = top_users[['user_id','total_orders','avg_days','reorder_rate']]
    top_users = top_users.rename(columns={'user_id':'User ID','total_orders':'Total Orders','avg_days':'Avg Days Between Orders','reorder_rate':'Reorder Rate'})
    top_users['Reorder Rate'] = (top_users['Reorder Rate']*100).round(1).astype(str)+'%'
    st.dataframe(top_users.reset_index(drop=True))

# -----------------------------
# Tab 3: Frequently Bought Together
# -----------------------------
with tabs[3]:
    st.header("Frequently Bought Together 🛒")
    st.write("See which products are commonly purchased together.")

    if not basket_rules.empty:
        # Convert frozensets to readable strings
        basket_rules['antecedents'] = basket_rules['antecedents'].apply(lambda x: ', '.join(eval(x)) if isinstance(x,str) else ', '.join(x))
        basket_rules['consequents'] = basket_rules['consequents'].apply(lambda x: ', '.join(eval(x)) if isinstance(x,str) else ', '.join(x))

        # Remove duplicates like A->B and B->A
        basket_rules['pair_sorted'] = basket_rules.apply(lambda row: ' 🛒 '.join(sorted([row['antecedents'],row['consequents']])), axis=1)
        basket_rules_unique = basket_rules.drop_duplicates(subset='pair_sorted')
        basket_rules_unique['times_bought_together'] = (basket_rules_unique['support']*10000).astype(int)

        top_rules = basket_rules_unique.sort_values('times_bought_together',ascending=False).head(10)
        top_rules = top_rules[['pair_sorted','times_bought_together']].rename(columns={'pair_sorted':'Products Bought Together','times_bought_together':'Times Bought Together'}).reset_index(drop=True)

        st.subheader("Top Product Pairs")
        cols = st.columns(2)
        for i,row in top_rules.iterrows():
            col = cols[i%2]
            max_val = top_rules['Times Bought Together'].max()
            intensity = int(255-(row['Times Bought Together']/max_val)*100)
            bg_color = f'rgb(135,{intensity},200)'
            with col:
                st.markdown(f"""
                <div style="background-color:{bg_color}; padding:20px; border-radius:12px; color:white; text-align:center;
                            box-shadow:2px 2px 8px rgba(0,0,0,0.25); margin-bottom:15px;">
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