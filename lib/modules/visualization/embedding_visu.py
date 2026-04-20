import numpy as np
import matplotlib.pyplot as plt

from lib.models.module import DiarizationModule
from lib.test.test_vad_output import play_segments
from lib.const import SAMPLING_RATE


class EmbeddingVisualizationModule(DiarizationModule):
    def __init__(
        self,
        labels: np.ndarray,
        data_plot: np.ndarray,
        wav_segments: list[np.ndarray] | None = None,
    ):
        super().__init__(tag="Embedding Visualization")
        self.labels = np.asarray(labels)
        self.data_plot = np.asarray(data_plot)
        self.wav_segments = wav_segments or []

    def run(self):
        if self.labels.size == 0 or self.data_plot.size == 0:
            print("No embeddings to visualize.")
            return None

        fig, ax = plt.subplots()
        unique_labels = sorted(set(self.labels))
        cmap = plt.cm.get_cmap("tab20", max(1, len(unique_labels)))
        artist_to_indices: dict[object, np.ndarray] = {}

        for i, lbl in enumerate(unique_labels):
            indices = np.where(self.labels == lbl)[0]
            color = "#9e9e9e" if lbl == -1 else cmap(i)
            name = "noise" if lbl == -1 else f"cluster {lbl}"
            artist = ax.scatter(
                self.data_plot[indices, 0],
                self.data_plot[indices, 1],
                s=40,
                alpha=0.9,
                c=[color],
                label=f"{name} ({len(indices)})",
                picker=True,
                pickradius=5,
            )
            artist_to_indices[artist] = indices

        def on_pick(event):
            artist = event.artist
            if artist not in artist_to_indices or len(event.ind) == 0:
                return

            idx = int(artist_to_indices[artist][event.ind[0]])
            print(idx)
            if idx < len(self.wav_segments):
                play_segments([self.wav_segments[idx]], sample_rate=SAMPLING_RATE)

        fig.canvas.mpl_connect("pick_event", on_pick)

        ax.legend(loc="lower right", title="Cluster")
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.show()

        return self.labels
