import numpy as np

# Paths to your files
ref_path = "data/data.npz"
mine_path = "data/E003_H3K4me1_deephistone_chr22_expected_format.npz"

# Load both
ref = np.load(ref_path)
mine = np.load(mine_path)

print("🔑 Comparing keys:")
ref_keys = set(ref.files)
mine_keys = set(mine.files)

print(f"Reference keys: {ref_keys}")
print(f"My keys       : {mine_keys}")

missing_in_mine = ref_keys - mine_keys
extra_in_mine = mine_keys - ref_keys

if missing_in_mine:
    print(f"⚠️  Missing in your file: {missing_in_mine}")
if extra_in_mine:
    print(f"⚠️  Extra keys in your file: {extra_in_mine}")

print("\n📐 Comparing shapes & values:\n")
for key in sorted(ref_keys & mine_keys):
    ref_arr = ref[key]
    mine_arr = mine[key]
    print(f"🔸 Key: {key}")
    print(f"  Shape: reference = {ref_arr.shape}, mine = {mine_arr.shape}")
    
    if ref_arr.shape != mine_arr.shape:
        print("  ❌ Shape mismatch\n")
        continue

    # Compare values
    all_close = np.allclose(ref_arr, mine_arr, atol=1e-5)
    mean_diff = np.mean(np.abs(ref_arr - mine_arr))
    max_diff = np.max(np.abs(ref_arr - mine_arr))
    print(f"  Values match: {all_close}")
    print(f"  Mean abs diff: {mean_diff:.5f}")
    print(f"  Max abs diff: {max_diff:.5f}\n")
