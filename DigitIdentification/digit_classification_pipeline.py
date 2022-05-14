# Import External Modules
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from gtda.plotting import plot_heatmap
from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, HeatKernel, Amplitude

# Defining the Pipeline
steps = [
    ("binarizer", Binarizer(threshold=0.4)),
    ("filtration", RadialFiltration(center=np.array([20, 6]))),
    ("diagram", CubicalPersistence()),
    ("rescaling", Scaler()),
    ("amplitude", Amplitude(metric="heat",
                            metric_params={'sigma': 0.15, 'n_bins': 60}))
]

heat_pipeline = Pipeline(steps)

# Import and format data
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.to_numpy()
X = X.reshape((-1, 28, 28))

# Select example and run through pipeline
img_idx = np.flatnonzero(y == "8")[0]
img = X[img_idx][None, :, :]
img_pipeline = heat_pipeline.fit_transform(img)
print(img_pipeline)
