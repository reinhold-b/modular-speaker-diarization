import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from lib.models.module import DiarizationModule

class CSpaceImprovedClusteringModule(DiarizationModule):
    def __init__(self, embeddings: list, n_clusters: int = None):
        super().__init__(tag="C-Space Improved Clustering")
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        
    def _estimate_num_clusters(self, X: np.ndarray, max_clusters: int = 5) -> int:
        """Estimate optimal number of clusters using silhouette score"""
        from sklearn.metrics import silhouette_score
        
        scores = []
        for n in range(2, min(max_clusters + 1, X.shape[0])):
            clustering = AgglomerativeClustering(
                n_clusters=n,
                metric="cosine",
                linkage="average",
            ).fit(X)
            score = silhouette_score(X, clustering.labels_, metric="cosine")
            scores.append((n, score))
            print(f"[DEBUG] n_clusters={n}: silhouette_score={score:.4f}")
        
        best_n = max(scores, key=lambda x: x[1])[0]
        print(f"[DEBUG] Estimated optimal clusters: {best_n}")
        return best_n

    def run(self):
        X = np.asarray(self.embeddings, dtype=np.float32)
        print(f"[DEBUG] Embeddings shape: {X.shape}")

        # Normalize
        X_normalized = normalize(X)

        # Determine number of clusters
        if self.n_clusters is None:
            n_clusters = self._estimate_num_clusters(X_normalized)
        else:
            n_clusters = self.n_clusters

        print(f"[DEBUG] Using {n_clusters} clusters")

        # Clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        ).fit(X_normalized)

        labels = clustering.labels_

        # Print cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"[DEBUG] Cluster distribution:")
        for cluster_id, n in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {n} embeddings ({100*n/len(labels):.1f}%)")

        # Visualization
        X_pca = PCA(n_components=2, random_state=42).fit_transform(X_normalized)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50, alpha=0.6)
        ax.set_title(f'Speaker Clustering ({n_clusters} clusters)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
        plt.savefig('/tmp/clustering.png', dpi=100)
        print(f"[DEBUG] Saved clustering plot to /tmp/clustering.png")
        plt.close()

        return labels, X_pca