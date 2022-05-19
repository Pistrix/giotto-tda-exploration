# Import External Modules
import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, Amplitude
from sklearn.ensemble import RandomForestClassifier


# Defining the Pipeline
steps = [
    ("binarizer", Binarizer(threshold=0.4)),
    ("filtration", RadialFiltration(center=np.array([20, 6]))),
    ("diagram", CubicalPersistence(reduced_homology=(False))),
    ("rescaling", Scaler()),
    ("amplitude", Amplitude(metric="heat",
                            metric_params={'sigma': 0.15, 'n_bins': 60})),
    ("classifier", RandomForestClassifier(random_state=(42)))
]

heat_pipeline = Pipeline(steps)

# Import and format data
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.to_numpy()
X = X.reshape((-1, 28, 28))
train_size, test_size = 1000, 50
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=666
)

# Fit and score pipeline
heat_pipeline.fit(X_train, y_train)
print(heat_pipeline.score(X_test, y_test))

filename_full = 'model_pipeline.sav'
pickle.dump(heat_pipeline, open(filename_full, 'wb'))
