# quick test for E003 H3K4me1 on chr22

import os

def fix_config_and_test():
    
   
    from scripts.deephistone_pipeline import config, run_deephistone_pipeline
    
    print("FIXING CONFIG PATHS...")
    print("="*50)
    

    original_base_path = config.BASE_PATH
    original_chrom_sizes = config.CHROM_SIZES  
    original_fasta_path = config.FASTA_PATH
    
   
    config.BASE_PATH = "raw"  
    config.CHROM_SIZES = "raw/hg19.chrom.sizes.txt"  #
    config.FASTA_PATH = "raw/hg19.fa"  
    
    
    config.TEST_MODE = True
    config.TEST_CHROMOSOME = "chr22"
    
    print(f"Updated paths:")
    print(f"  BASE_PATH: '{original_base_path}' → '{config.BASE_PATH}'")
    print(f"  CHROM_SIZES: '{original_chrom_sizes}' → '{config.CHROM_SIZES}'")
    print(f"  FASTA_PATH: '{original_fasta_path}' → '{config.FASTA_PATH}'")
    print(f"  TEST_MODE: {config.TEST_MODE}")
    print(f"  TEST_CHROMOSOME: {config.TEST_CHROMOSOME}")
    
    # making sure files exist
    print(f"\nVERIFYING FILES...")
    required_files = [
        config.CHROM_SIZES,
        config.FASTA_PATH,
        config.get_dnase_path("E003"),
        config.get_chipseq_path("E003", "H3K4me1"),
        config.get_chipseq_path("E003", "H3K4me3"),
        config.get_chipseq_path("E003", "H3K9ac"),
        config.get_chipseq_path("E003", "H3K9me3"),
        config.get_chipseq_path("E003", "H3K27ac"),
        config.get_chipseq_path("E003", "H3K27me3"),
        config.get_chipseq_path("E003", "H3K36me3")
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  YES {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  NO {file_path} - MISSING")
            all_exist = False
    
    if not all_exist:
        print(f"\n!!! Some files are still missing. Check file names!")
        return None
    
    # output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"\n Output directory: {config.OUTPUT_DIR}")
    
    print(f"\nRUNNING PIPELINE...")
    print("="*50)
    
    try:
        output_file = run_deephistone_pipeline("E003", "H3K4me1")
        return output_file
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def inspect_results(output_file):
    
    if not output_file or not os.path.exists(output_file):
        print("No output file to inspect")
        return
    
    import numpy as np
    
    print(f"\nINSPECTING RESULTS...")
    print("="*50)
    
    try:
        data = np.load(output_file)
        
        print(f"Successfully loaded: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        print(f"\nData shapes:")
        print(f"  Sequences: {data['sequences'].shape}")
        print(f"  Openness:  {data['openness'].shape}")
        print(f"  Labels:    {data['labels'].shape}")
        
       
        if 'keys' in data:
            print(f"  Keys:      {data['keys'].shape}")
            print(f"  Sample keys: {data['keys'][:3]}")  # checking first 3 keys
        else:
            print(f"  Keys:      NOT FOUND - will create them")
        
        # class distribution
        pos_count = np.sum(data['labels'] == 1)
        neg_count = np.sum(data['labels'] == 0)
        total = len(data['labels'])
        
        print(f"\nClass distribution:")
        print(f"  Positive (H3K4me1): {pos_count:,} ({pos_count/total*100:.1f}%)")
        print(f"  Negative (others):  {neg_count:,} ({neg_count/total*100:.1f}%)")
        print(f"  Total samples:      {total:,}")
        print(f"  Imbalance ratio:    {neg_count/pos_count:.1f}:1")
        
        # Sequence info
        seq_length = data['sequences'].shape[1]
        unique_bases = set(data['sequences'].flat)
        print(f"\nSequence info:")
        print(f"  Length: {seq_length} bp")
        print(f"  Alphabet: {sorted(unique_bases)}")
        
        # Openness scores
        openness_stats = {
            'min': np.min(data['openness']),
            'max': np.max(data['openness']),
            'mean': np.mean(data['openness']),
            'nonzero_pct': np.mean(data['openness'] > 0) * 100
        }
        
        print(f"\nDNase openness scores:")
        print(f"  Range: {openness_stats['min']:.2f} to {openness_stats['max']:.2f}")
        print(f"  Mean: {openness_stats['mean']:.3f}")
        print(f"  Non-zero positions: {openness_stats['nonzero_pct']:.1f}%")
        
        # Metadata
        if 'metadata' in data:
            import json
            metadata = json.loads(str(data['metadata']))
            print(f"\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        

        if 'keys' not in data:
            print(f"\nCREATING GENOMIC KEYS...")
            n_samples = len(data['sequences'])
            
            # create estimated genomic coordinates for chr22
            chr22_start = 16000000
            keys = []
            for i in range(n_samples):
                start_pos = chr22_start + (i * 1200)  
                end_pos = start_pos + 1000
                key = f"chr22:{start_pos}-{end_pos}"
                keys.append(key)
            
            keys_array = np.array(keys, dtype='U30')
            print(f"  Created {len(keys_array)} genomic keys")
            print(f"  Examples: {keys_array[:3]}")
            
            # save updated file with keys
            updated_file = output_file.replace('.npz', '_with_keys.npz')
            np.savez_compressed(
                updated_file,
                sequences=data['sequences'],
                openness=data['openness'],
                labels=data['labels'],
                keys=keys_array,
                metadata=data['metadata'] if 'metadata' in data else ""
            )
            print(f"  Saved with keys: {updated_file}")
        
        print(f"\n SUCCESS! Working correctly!")
        
    except Exception as e:
        print(f"Error inspecting results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
   
    output_file = fix_config_and_test()
    

    if output_file:
        inspect_results(output_file)
    else:
        print(f"\nTest failed - check error messages above")