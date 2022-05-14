## Module imports
# Step 1
from sklearn.datasets import fetch_openml

# Step 2
import numpy as np
from gtda.plotting import plot_heatmap  # for plotting data

# Step 3
from sklearn.model_selection import train_test_split

# Step 4
from gtda.images import Binarizer

# Step 5
from gtda.images import RadialFiltration

# Step 6
from gtda.homology import CubicalPersistence

# Step 7
from gtda.diagrams import Scaler

# Step 8
from gtda.diagrams import HeatKernel

## Step 1: Import and examine the dataset
# X stores 70,000 images where each image has 784 (28*28) features
# y stores the classifications
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
print(f"X shape: {X.shape}, y shape: {y.shape}")

## Step 2: Visualize the raw data with a heatmap function
X = X.to_numpy()
img7_idx = np.flatnonzero(y == "7")[0]  # Grab the first example of a "7"
img7 = np.reshape(X[img7_idx], (28, 28))
#plot_heatmap(img7).show()

## Step 3: Create train and test sets
train_size, test_size = 60, 10

X = X.reshape((-1, 28, 28))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=666
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Step 4: Binarize the image to align with gtda filtrations
img7_idx = np.flatnonzero(y_train == "8")[0]
img7_heatmap = X_train[img7_idx]
img7 = X_train[img7_idx][None, :, :]

binarizer = Binarizer(threshold=0.4)
img7_binarized = binarizer.fit_transform(img7)

#plot_heatmap(img7_heatmap).show()
#binarizer.plot(img7_binarized).show()

# Step 5: Apply radial filtration to the binarized image
radial_filtration = RadialFiltration(center=np.array([20, 6]))
img7_filtration = radial_filtration.fit_transform(img7_binarized)

#radial_filtration.plot(img7_filtration, colorscale="jet").show()

# Step 6: Convert from filtration to persistence diagram
cubical_persistence = CubicalPersistence(n_jobs=-1)
img7_cubical = cubical_persistence.fit_transform(img7_filtration)

#cubical_persistence.plot(img7_cubical).show()

# Step 7: Postprocess the persistence diagram
scaler = Scaler()
img7_scaled = scaler.fit_transform(img7_cubical)

#scaler.plot(img7_scaled).show()

# Step 8: Convert the scaled persitence diagram to a representation
heat = HeatKernel(sigma=.15, n_bins=60, n_jobs=-1)
img7_heat = heat.fit_transform(img7_scaled)

heat.plot(img7_heat, homology_dimension_idx=1, colorscale="jet").show()
