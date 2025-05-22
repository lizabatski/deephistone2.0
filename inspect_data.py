import numpy as np

# Load the .npz file
data = np.load("data/data.npz")

# List all keys
print("Keys found in data.npz:", data.files)

# Print shape and type for each array
for key in data.files:
    print(f"\nKey: {key}")
    print(f"Shape: {data[key].shape}")
    print(f"Dtype: {data[key].dtype}")
    print(f"First 5 entries:\n{data[key][:5]}")
