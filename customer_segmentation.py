import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_superstore.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

df["Order Date"] = pd.to_datetime(df["Order Date"])

df = df.drop_duplicates()

df = df.dropna(subset=[
    "Customer ID",
    "Order ID",
    "Order Date",
    "Sales",
    "Profit",
    "Quantity"
])

snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

customer_data = df.groupby("Customer ID").agg(
    Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Order ID", "nunique"),
    Monetary=("Sales", "sum"),
    Profit=("Profit", "sum"),
    Quantity=("Quantity", "sum")
).reset_index()

print("\nCustomer-level data created!")
print("Number of customers:", len(customer_data))

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Profit",
    "Quantity"
]

X = customer_data[features].copy()

# KMeans requires finite numeric values. Some features (for example profit or sales)
# can be negative because of refunds or returns, which makes log1p() produce NaN.
# Shift each feature so its minimum is zero before taking the log transform.
X = X - X.min()
X = X.clip(lower=0)
X_log = np.log1p(X)

if not np.isfinite(X_log.to_numpy()).all():
    raise ValueError("Feature matrix contains non-finite values after preprocessing.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    
    print(f"K={k}, Silhouette Score={score:.4f}")

optimal_k = range(2, 11)[np.argmax(silhouette_scores)]

print("\nOptimal number of clusters:", optimal_k)

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = customer_data.groupby("Cluster").agg(
    Customers=("Customer ID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Avg_Profit=("Profit", "mean"),
    Avg_Quantity=("Quantity", "mean")
).reset_index()

print("\nCluster Summary:")
print(cluster_summary)

customer_data.to_csv(
    "customer_segments.csv",
    index=False
)

cluster_summary.to_csv(
    "cluster_summary.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for K-Means Clustering")
plt.xticks(range(2, 11))
plt.grid(True)
plt.savefig("silhouette_scores.png")
plt.show()

print("\nFiles created:")
print("customer_segments.csv")
print("cluster_summary.csv")
print("silhouette_scores.png")

print("\nCustomer Segment Analysis")

for _, row in cluster_summary.iterrows():
    print(f"\nCluster {int(row['Cluster'])}")
    print(f"Customers: {int(row['Customers'])}")
    print(f"Average Recency: {row['Avg_Recency']:.2f}")
    print(f"Average Frequency: {row['Avg_Frequency']:.2f}")
    print(f"Average Monetary: {row['Avg_Monetary']:.2f}")
    print(f"Average Profit: {row['Avg_Profit']:.2f}")
    print(f"Average Quantity: {row['Avg_Quantity']:.2f}")