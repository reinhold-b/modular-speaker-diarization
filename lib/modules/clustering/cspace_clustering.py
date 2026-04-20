import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

from lib.models.module import DiarizationModule

class CSpaceClusteringModule(DiarizationModule):
    def __init__(self, embeddings: list):
        super().__init__(tag="C-Space Clustering")
        self.embeddings = embeddings

    def run(self):
        X = np.asarray(self.embeddings, dtype=np.float32)

        # 1) Dim-Reduktion vor Clustering (stabiler)
        #n_comp = min(50, X.shape[1], X.shape[0] - 1)
        #X = PCA(n_components=max(2, n_comp), random_state=42).fit_transform(X)

        data_scaled = normalize(X)

        clustering = AgglomerativeClustering(
            n_clusters=4,
            metric="cosine",
            linkage="average",
        ).fit(data_scaled)

        labels = clustering.labels_

        with open("cspace_labels.csv", "w") as f:
            unique, counts = np.unique(labels, return_counts=True)
            for cluster_id, n in zip(unique, counts):
                print(f"Cluster {cluster_id}: {n} samples")
                f.write(f"{cluster_id},{n}\n")

        data_plot = PCA(n_components=2, random_state=42).fit_transform(data_scaled)
        
        return labels, data_plot
