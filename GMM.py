# Clustering Iris dataset using Gaussian Mixture Model (GMM)
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data  # using all 4 features

# Create GMM model
gmm = GaussianMixture(n_components=3, random_state=42)

# Fit the model
gmm.fit(X)

# Predict cluster labels
cluster_labels = gmm.predict(X)

# Since clustering is unsupervised, we can compare with true labels (optional)
true_labels = iris.target

# Sometimes clustering labels and true labels don't match directly
# So we can adjust them for accuracy (simple matching here)
# But for now, let's just see cluster labels

print("Cluster labels assigned by GMM:")
print(cluster_labels)
