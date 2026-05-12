# ============================================================
# CUSTOMER SEGMENTATION DASHBOARD - STREAMLIT
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.title("🛍️ Customer Segmentation Dashboard")
st.write("Analyze customer groups using K-Means Clustering")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customer.csv")
    return df

df = load_data()

# ------------------------------------------------------------
# REMOVE MISSING VALUES
# ------------------------------------------------------------

df.dropna(inplace=True)

# ------------------------------------------------------------
# DISPLAY DATA
# ------------------------------------------------------------

st.subheader("📄 Customer Dataset")

st.dataframe(df)

# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Average Age", round(df['Age'].mean(), 1))

with col3:
    st.metric(
        "Average Income",
        round(df['Annual Income (k$)'].mean(), 1)
    )

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("⚙️ Clustering Settings")

clusters = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
    value=5
)

# ------------------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------------------

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# K-MEANS MODEL
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=clusters,
    random_state=42
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------------------
# CUSTOMER SEGMENTS VISUALIZATION
# ------------------------------------------------------------

st.subheader("🎯 Customer Segments")

fig, ax = plt.subplots(figsize=(10,6))

scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    s=100
)

# Cluster Centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

ax.scatter(
    centers[:,0],
    centers[:,1],
    s=300,
    marker='X'
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segmentation")

st.pyplot(fig)

# ------------------------------------------------------------
# CLUSTER SUMMARY
# ------------------------------------------------------------

st.subheader("📈 Cluster Summary")

cluster_summary = df.groupby('Cluster').mean(numeric_only=True)

st.dataframe(cluster_summary)

# ------------------------------------------------------------
# CUSTOMER DISTRIBUTION
# ------------------------------------------------------------

st.subheader("👥 Customers in Each Cluster")

cluster_counts = df['Cluster'].value_counts().sort_index()

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax2.set_xlabel("Cluster")
ax2.set_ylabel("Number of Customers")
ax2.set_title("Cluster Distribution")

st.pyplot(fig2)

# ------------------------------------------------------------
# BUSINESS INSIGHTS
# ------------------------------------------------------------

st.subheader("💡 Business Insights")

st.markdown("""
### Insights from Customer Segmentation

- Customers with high income and high spending are premium customers.
- Customers with low income and high spending may respond well to discounts.
- Customers with high income but low spending can be targeted with marketing campaigns.
- Personalized recommendations can improve customer engagement.
- Businesses can use segmentation for targeted advertising and sales strategies.
""")

# ------------------------------------------------------------
# DOWNLOAD OUTPUT
# ------------------------------------------------------------

st.subheader("⬇️ Download Clustered Data")

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV File",
    data=csv,
    file_name='Customer_Segments_Output.csv',
    mime='text/csv'
)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")
st.markdown("Customer Segmentation Project using Python & Streamlit")