import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Generate sample data
np.random.seed(0)  # For reproducibility

# Sample size
n_samples = 1000

# Features: weight (kg), height (cm), leg length (cm), fur length (cm)
# Weights: 2-10 kg, Heights: 20-70 cm, Leg Lengths: 5-25 cm, Fur Lengths: 1-10 cm
weights = np.random.uniform(2, 10, n_samples)
heights = np.random.uniform(20, 70, n_samples)
leg_lengths = np.random.uniform(5, 25, n_samples)
fur_lengths = np.random.uniform(1, 10, n_samples)

# Fur color: Black, Brown, White, Grey, Spotted (one-hot encoded)
fur_colors = np.random.choice(['Black', 'Brown', 'White', 'Grey', 'Spotted'], n_samples)
encoder = OneHotEncoder(sparse=False)
fur_colors_encoded = encoder.fit_transform(fur_colors.reshape(-1, 1))

# Labels (0 for cat, 1 for dog)
labels = np.random.randint(0, 2, n_samples)

# Combine all features
features = np.column_stack((weights, heights, leg_lengths, fur_lengths, fur_colors_encoded))

# Split into features (X) and labels (y)
X = features
y = labels

X.shape, y.shape, X[:5], y[:5]  # Show the shape of the data and the first 5 samples

