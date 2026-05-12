# ============================================================
# CUSTOMER SEGMENTATION PROJECT (Fixed for NaN values)
# ============================================================

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Step 2: Load Dataset
df = pd.read_csv("Mall_Customer.csv")  # use your actual file name

print("First 5 Rows of Dataset:")
print(df.head())

# Step 3: Basic Information
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Exploratory Data Analysis (EDA)
sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Annual Income (k$)'], bins=20)
plt.title("Annual Income Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Spending Score (1-100)'], bins=20)
plt.title("Spending Score Distribution")
plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    data=df
)
plt.title("Income vs Spending Score")
plt.show()

# Step 5: Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print("\nSelected Features:\n", X.head())

# ✅ Step 6: Handle Missing Values
imputer = SimpleImputer(strategy="mean")  # replace NaN with column mean
X_imputed = imputer.fit_transform(X)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print("\nScaled Features:\n", X_scaled[:5])

# Step 8: Find Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 9: Train K-Means Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
df['Cluster'] = y_kmeans

print("\nClustered Data:\n", df.head())

# Step 10: Visualize Customer Segments
plt.figure(figsize=(10,7))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    data=df,
    s=100
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:,0],
    centers[:,1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()

# Step 11: Cluster Analysis
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print("\nCluster Summary:\n", cluster_summary)

# Step 12: Save Output File
df.to_csv("Customer_Segments_Output.csv", index=False)
print("\nCustomer Segmentation Completed Successfully!")
print("Output file saved as Customer_Segments_Output.csv")
