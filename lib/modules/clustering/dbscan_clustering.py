import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from lib.models.module import DiarizationModule

class DBSCANClusteringModule(DiarizationModule):
    def __init__(self, embeddings: list):
        super().__init__(tag="DBSCAN Clustering")
        self.embeddings = embeddings

    def run(self):
        # This is a placeholder implementation. You would need to implement the actual logic
        # to perform DBSCAN clustering on the embeddings and return the cluster labels.
        scaler = StandardScaler().fit(self.embeddings)
        data_scaled = scaler.transform(self.embeddings)
        data_scaled = normalize(data_scaled)

        # Reduziere die Dimensionen massiv (z.B. auf 5), damit die Dichte von Punkten greifen kann
        pca_cluster = PCA(n_components=min(5, data_scaled.shape[0], data_scaled.shape[1]))
        data_cluster = pca_cluster.fit_transform(data_scaled)

        cluster = DBSCAN(eps=0.3, min_samples=5, metric="cosine").fit(data_cluster)
        labels = cluster.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        print(f"clusters={n_clusters}, noise={n_noise}")

        pca_plot = PCA(n_components=2, random_state=42)
        data_plot = pca_plot.fit_transform(data_scaled)

        return labels, data_plot

