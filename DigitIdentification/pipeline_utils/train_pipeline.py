import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.ensemble import RandomForestClassifier
from gtda.images import Binarizer, RadialFiltration, HeightFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, Amplitude, PersistenceEntropy

direction_list = [[1, 0], [1, 1], [0, 1],
                  [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

center_list = [
    [13, 6],
    [6, 13],
    [13, 13],
    [20, 13],
    [13, 20],
    [6, 6],
    [6, 20],
    [20, 6],
    [20, 20],
]

# Creating a list of all filtration transformer, we will be applying
filtration_list = (
    [
        HeightFiltration(direction=np.array(direction), n_jobs=-1)
        for direction in direction_list
    ]
    + [RadialFiltration(center=np.array(center), n_jobs=-1)
       for center in center_list]
)

# Creating the diagram generation pipeline
diagram_steps = [
    [
        Binarizer(threshold=0.4, n_jobs=-1),
        filtration,
        CubicalPersistence(n_jobs=-1, reduced_homology=(False)),
        Scaler(n_jobs=-1),
    ]
    for filtration in filtration_list
]

# Listing all metrics we want to use to extract diagram amplitudes
metric_list = [
    {"metric": "bottleneck", "metric_params": {}},
    {"metric": "wasserstein", "metric_params": {"p": 1}},
    {"metric": "wasserstein", "metric_params": {"p": 2}},
    {"metric": "landscape", "metric_params": {
        "p": 1, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {
        "p": 1, "n_layers": 2, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {
        "p": 2, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {
        "p": 2, "n_layers": 2, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
]

#
feature_union = make_union(
    *[PersistenceEntropy(nan_fill_value=-1)]
    + [Amplitude(**metric, n_jobs=-1) for metric in metric_list]
)

tda_union = make_union(
    *[make_pipeline(*diagram_step, feature_union)
      for diagram_step in diagram_steps],
    n_jobs=-1
)

model = make_pipeline(
    tda_union,
    RandomForestClassifier(n_estimators=500)
)

## Step 1: Import and examine the dataset
# X stores 70,000 images where each image has 784 (28*28) features, y stores the features
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.to_numpy()
train_size, test_size = 2000, 300
X = X.reshape((-1, 28, 28))

X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=666
        )

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

filename_model = 'data/model_pipeline.sav'
pickle.dump(model, open(filename_model, 'wb'))
