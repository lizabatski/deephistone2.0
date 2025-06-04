import numpy as np
import os

def convert_to_expected_format(input_file):
    """Convert ALL samples from pipeline output to expected format"""
    
    print(f" Converting {input_file}...")
    
    if not os.path.exists(input_file):
        print(f" File not found: {input_file}")
        return None
    
    # Load your pipeline output
    data = np.load(input_file)
    sequences = data['sequences']
    openness = data['openness'] 
    labels = data['labels']
    
    # Get the actual number of samples
    n_samples = len(sequences)
    print(f" Processing ALL {n_samples:,} samples...")
    
    # Get keys if they exist, otherwise create them
    if 'keys' in data.files:
        keys = data['keys']
        print(f" Using existing genomic keys")
        print(f"   Examples: {keys[:3].tolist()}")
    else:
        keys = np.array([f"region_{i:06d}" for i in range(n_samples)], dtype='U20')
        print(f"  Created generic keys")
    
    # Convert DNA to one-hot encoding
    print(f" Converting {n_samples:,} DNA sequences to one-hot...")
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    dna_onehot = np.zeros((n_samples, 1, 4, 1000), dtype=np.float32)
    
    for i in range(n_samples):
        if i % 1000 == 0:  # Progress indicator
            print(f"   Processing sequence {i:,}/{n_samples:,}")
        
        # Handle different sequence formats
        seq = sequences[i]
        if isinstance(seq, str):
            seq_str = seq
        else:
            seq_str = ''.join(seq)  # Convert character array to string
        
        # Convert to one-hot
        for j, base in enumerate(seq_str[:1000]):  # Ensure max 1000bp
            base_idx = base_to_idx.get(base.upper(), 0)
            dna_onehot[i, 0, base_idx, j] = 1.0
    
    print(f" DNA conversion complete: {dna_onehot.shape}")
    
    # Reshape DNase data
    print(f" Reshaping {n_samples:,} DNase accessibility arrays...")
    dnase = openness.reshape(n_samples, 1, 1, 1000).astype(np.float32)
    print(f" DNase reshape complete: {dnase.shape}")
    
    # Create multi-task labels (7 histone markers)
    print(f" Creating multi-task labels for {n_samples:,} samples...")
    label = np.zeros((n_samples, 1, 7), dtype=np.float32)
    # H3K4me1 is at index 1 in ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K9ac']
    label[:, 0, 1] = labels.astype(np.float32)
    print(f" Multi-task labels complete: {label.shape}")
    
    # Save in expected format
    output_file = input_file.replace('.npz', '_expected_format.npz')
    print(f" Saving {n_samples:,} samples to {output_file}...")
    
    np.savez_compressed(
        output_file, 
        keys=keys, 
        dna=dna_onehot, 
        dnase=dnase, 
        label=label
    )
    
    # Verify and show results
    print(f"Conversion complete!")
    print(f"\n Final format:")
    print(f"   keys:  {keys.shape} - {keys.dtype}")
    print(f"   dna:   {dna_onehot.shape} - {dna_onehot.dtype}")
    print(f"   dnase: {dnase.shape} - {dnase.dtype}")
    print(f"   label: {label.shape} - {label.dtype}")
    
    # Show file size
    file_size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"\n File: {output_file}")
    print(f"   Size: {file_size_mb:.1f} MB")
    
    # Show sample data
    pos_samples = np.sum(labels)
    print(f"\n Dataset stats:")
    print(f"   Total samples: {n_samples:,}")
    print(f"   Positive (H3K4me1): {pos_samples:,} ({pos_samples/n_samples*100:.1f}%)")
    print(f"   Negative: {n_samples-pos_samples:,} ({(n_samples-pos_samples)/n_samples*100:.1f}%)")
    
    # Verify one-hot encoding
    sample_sums = np.sum(dna_onehot[:10], axis=2)  # Check first 10 samples
    print(f"   One-hot check: {sample_sums.min():.1f} to {sample_sums.max():.1f} (should be 1.0)")
    
    return output_file

def quick_verify(npz_file):
    """Quick verification of the converted file"""
    
    print(f"\n QUICK VERIFICATION: {npz_file}")
    print("="*50)
    
    if not os.path.exists(npz_file):
        print(f" File not found")
        return
    
    data = np.load(npz_file)
    
    # Check expected format
    expected = {
        'keys': 'string identifiers',
        'dna': '(N, 1, 4, 1000) one-hot DNA',  
        'dnase': '(N, 1, 1000) accessibility',
        'label': '(N, 1, 7) multi-task labels'
    }
    
    print("Format verification:")
    for key, description in expected.items():
        if key in data.files:
            shape = data[key].shape
            dtype = data[key].dtype
            print(f"   {key:6}: {shape} {dtype} - {description}")
        else:
            print(f"   {key:6}: MISSING")
    
    # Sample data
    if 'keys' in data.files:
        print(f"\nSample data:")
        print(f"  keys[0]: {data['keys'][0]}")
        print(f"  dna[0] sum per position: {np.sum(data['dna'][0], axis=0)[:5]}")
        print(f"  dnase[0] range: {data['dnase'][0].min():.3f} to {data['dnase'][0].max():.3f}")
        print(f"  label[0]: {data['label'][0]}")

if __name__ == "__main__":
    # Convert your pipeline output - ALL SAMPLES
    input_file = "data/E003_H3K4me1_deephistone_chr22.npz"
    
    if os.path.exists(input_file):
        output_file = convert_to_expected_format(input_file)
        
        if output_file:
            quick_verify(output_file)
            print(f"\ SUCCESS! All {len(np.load(input_file)['sequences']):,} samples converted!")
        else:
            print(f"Conversion failed")
    else:
        print(f"Input file not found: {input_file}")
        print("Make sure your DeepHistone pipeline completed successfully.")
        
        # Show available files
        print(f"\nAvailable files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.npz'):
                print(f"   {f}")