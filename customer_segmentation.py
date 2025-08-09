# customer_segmentation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline

# ----------------------
# 1. Load dataset
# ----------------------
# Put the CSV in the same folder or provide full path
df = pd.read_csv("Mall_Customers.csv")  # adjust filename if needed

# ----------------------
# 2. Quick EDA
# ----------------------
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Basic plots
sns.countplot(x='Gender', data=df)
plt.title("Gender count")
plt.show()

# histograms
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].hist(bins=20, figsize=(10,4))
plt.tight_layout()
plt.show()

# ----------------------
# 3. Feature selection
# ----------------------
# Typical features used: Age, Annual Income (k$), Spending Score (1-100)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

# Optionally encode gender if you want to include it:
# df['Gender_Num'] = df['Gender'].map({'Male':0, 'Female':1})
# features = ['Age','Annual Income (k$)','Spending Score (1-100)','Gender_Num']

# ----------------------
# 4. Feature scaling
# ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------
# 5. Choose k using Elbow + Silhouette
# ----------------------
inertia = []
sil_scores = []
K = range(2,11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    sil_scores.append(sil)

# Plot elbow and silhouette
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(K, inertia, '-o')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1,2,2)
plt.plot(K, sil_scores, '-o')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.show()

# Choose k (inspect plots) - example choose k=5 (commonly used for this dataset)
k_opt = 5

# ----------------------
# 6. Fit KMeans
# ----------------------
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

df['cluster'] = labels

# cluster centers (in original feature scale)
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers, columns=features)
centers_df['cluster'] = range(k_opt)
print("Cluster centers (original scale):\n", centers_df)

# Silhouette for chosen k
print("Silhouette score for k =", k_opt, "->", silhouette_score(X_scaled, labels))

# ----------------------
# 7. Visualization
# ----------------------
# 7a. 2D via PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
palette = sns.color_palette('tab10', n_colors=k_opt)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette=palette, s=60, alpha=0.8)
for i, row in centers_df.iterrows():
    # project center to PCA space for plotting
    center_scaled = scaler.transform(row[features].values.reshape(1,-1))
    center_pca = pca.transform(center_scaled)
    plt.scatter(center_pca[0,0], center_pca[0,1], marker='X', s=200, c='black')
plt.title('KMeans clusters (PCA projection)')
plt.legend(title='cluster')
plt.show()

# 7b. Pairplot (Age vs Income vs Spending Score) colored by cluster
sns.pairplot(df, vars=features, hue='cluster', palette=palette, diag_kind='kde', height=2.5)
plt.show()

# 7c. Summary table
cluster_summary = df.groupby('cluster')[features].agg(['count','mean','median']).round(2)
print(cluster_summary)

# ----------------------
# 8. (Optional) Hierarchical clustering example
# ----------------------
agg = AgglomerativeClustering(n_clusters=k_opt, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)
df['agg_cluster'] = agg_labels

# Compare counts
print(df['agg_cluster'].value_counts())

# ----------------------
# 9. Business insights (printable)
# ----------------------
for c in sorted(df['cluster'].unique()):
    row = centers_df.loc[centers_df['cluster']==c, features].iloc[0]
    print(f"\nCluster {c}:")
    print(f" Age (avg): {row['Age']:.1f}, Annual Income (k$): {row['Annual Income (k$)']:.1f}, Spending Score: {row['Spending Score (1-100)']:.1f}")
