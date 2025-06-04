import numpy as np
import os
import json
import time
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging
from datetime import datetime
import traceback


class BatchConverterConfig:
    def __init__(self):
        # Input/output directories
        self.INPUT_DIR = "data"  # Where your pipeline outputs are
        self.OUTPUT_DIR = "data/converted"  # Where converted files go
        
        # Processing settings
        self.USE_MULTIPROCESSING = True
        self.N_PROCESSES = min(4, mp.cpu_count())  # Fewer processes for memory-intensive conversion
        self.BATCH_SIZE = 3  # Smaller batches for conversion (uses more memory)
        
        # File handling
        self.SKIP_EXISTING = True
        self.CONTINUE_ON_ERROR = True
        
        # The 7 histone markers in order
        self.ALL_MARKERS = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K9ac']
        
        # Test mode suffix (if you used test mode)
        self.TEST_MODE_SUFFIX = "_chr22"  # Set to "" if you didn't use test mode


config = BatchConverterConfig()


def setup_conversion_logging():
    """Set up logging for batch conversion"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/batch_conversion_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def find_pipeline_outputs():
    """Find all pipeline output files to convert"""
    pipeline_files = []
    
    if not os.path.exists(config.INPUT_DIR):
        print(f"Input directory not found: {config.INPUT_DIR}")
        return []
    
    # Look for files matching the pattern: EpigenomeID_Marker_deephistone[_chr22].npz
    for filename in os.listdir(config.INPUT_DIR):
        if filename.endswith('.npz') and 'deephistone' in filename and 'expected_format' not in filename:
            # Parse filename to extract epigenome and marker
            try:
                # Expected format: E003_H3K4me1_deephistone_chr22.npz or E003_H3K4me1_deephistone.npz
                parts = filename.replace('.npz', '').split('_')
                
                if len(parts) >= 3:
                    epigenome_id = parts[0]
                    marker = parts[1]
                    
                    if marker in config.ALL_MARKERS:
                        full_path = os.path.join(config.INPUT_DIR, filename)
                        pipeline_files.append((epigenome_id, marker, full_path, filename))
                        
            except Exception as e:
                print(f"Warning: Could not parse filename {filename}: {e}")
                continue
    
    return sorted(pipeline_files)


def get_marker_index(marker):
    """Get the index of a marker in the ALL_MARKERS list"""
    try:
        return config.ALL_MARKERS.index(marker)
    except ValueError:
        print(f"Warning: Unknown marker {marker}, using index 0")
        return 0


def convert_single_dataset(epigenome_id, marker, input_path, input_filename, logger=None):
    """Convert a single dataset to expected format"""
    start_time = time.time()
    
    try:
        if logger:
            logger.info(f"Starting conversion: {epigenome_id}-{marker}")
        
        print(f"\n{'='*60}")
        print(f"Converting: {epigenome_id}-{marker}")
        print(f"Input: {input_filename}")
        print(f"{'='*60}")
        
        # Check if output already exists
        output_filename = input_filename.replace('.npz', '_expected_format.npz')
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)
        
        if config.SKIP_EXISTING and os.path.exists(output_path):
            print(f"Output already exists, skipping: {output_filename}")
            if logger:
                logger.info(f"Skipped {epigenome_id}-{marker} - already exists")
            return output_path, True, "skipped"
        
        # Load pipeline output
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print("Loading pipeline output...")
        data = np.load(input_path, allow_pickle=True)
        
        # Validate required keys
        required_keys = ['sequences', 'openness', 'labels']
        for key in required_keys:
            if key not in data.files:
                raise ValueError(f"Missing required key in input file: {key}")
        
        sequences = data['sequences']
        openness = data['openness'] 
        labels = data['labels']
        
        n_samples = len(sequences)
        print(f"Processing {n_samples:,} samples...")
        
        if n_samples == 0:
            raise ValueError("No samples found in input file")
        
        # Get or create genomic keys
        if 'keys' in data.files:
            keys = data['keys']
            print(f"Using existing genomic keys")
            print(f"  Examples: {keys[:3].tolist()}")
        else:
            keys = np.array([f"region_{i:06d}" for i in range(n_samples)], dtype='U30')
            print(f"Created generic keys")
        
        # Convert DNA sequences to one-hot encoding
        print(f"Converting {n_samples:,} DNA sequences to one-hot...")
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        dna_onehot = np.zeros((n_samples, 1, 4, 1000), dtype=np.float32)
        
        # Process sequences in chunks to show progress
        chunk_size = max(1, n_samples // 10)  # 10 progress updates
        
        for i in range(n_samples):
            if i % chunk_size == 0:
                progress = (i / n_samples) * 100
                print(f"  DNA conversion progress: {progress:.1f}% ({i:,}/{n_samples:,})")
            
            # Handle different sequence formats
            seq = sequences[i]
            if isinstance(seq, str):
                seq_str = seq
            else:
                seq_str = ''.join(seq)  # Convert character array to string
            
            # Convert to one-hot (ensure max 1000bp)
            seq_len = min(len(seq_str), 1000)
            for j in range(seq_len):
                base = seq_str[j].upper()
                base_idx = base_to_idx.get(base, 0)  # Default to 'A' for unknown bases
                dna_onehot[i, 0, base_idx, j] = 1.0
        
        print(f"DNA conversion complete: {dna_onehot.shape}")
        
        # Reshape DNase accessibility data
        print(f"Reshaping {n_samples:,} DNase accessibility arrays...")
        if openness.ndim == 1:
            # If 1D, assume each element is an array
            dnase_arrays = []
            for i in range(n_samples):
                if isinstance(openness[i], np.ndarray):
                    dnase_array = openness[i][:1000]  # Take first 1000bp
                    if len(dnase_array) < 1000:
                        # Pad with zeros if shorter
                        padded = np.zeros(1000, dtype=np.float32)
                        padded[:len(dnase_array)] = dnase_array
                        dnase_array = padded
                    dnase_arrays.append(dnase_array)
                else:
                    # If not an array, create zeros
                    dnase_arrays.append(np.zeros(1000, dtype=np.float32))
            
            dnase = np.array(dnase_arrays).reshape(n_samples, 1, 1, 1000).astype(np.float32)
        else:
            # If already 2D, reshape directly
            dnase = openness.reshape(n_samples, 1, 1, 1000).astype(np.float32)
        
        print(f"DNase reshape complete: {dnase.shape}")
        
        # Create multi-task labels (7 histone markers)
        print(f"Creating multi-task labels for {n_samples:,} samples...")
        label = np.zeros((n_samples, 1, 7), dtype=np.float32)
        
        # Set the target marker at the correct index
        marker_idx = get_marker_index(marker)
        label[:, 0, marker_idx] = labels.astype(np.float32)
        
        print(f"Multi-task labels complete: {label.shape}")
        print(f"Target marker '{marker}' set at index {marker_idx}")
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # Save in expected format
        print(f"Saving {n_samples:,} samples to {output_filename}...")
        
        np.savez_compressed(
            output_path, 
            keys=keys, 
            dna=dna_onehot, 
            dnase=dnase, 
            label=label
        )
        
        # Verify the saved file
        verify_data = np.load(output_path)
        expected_keys = ['keys', 'dna', 'dnase', 'label']
        for key in expected_keys:
            if key not in verify_data.files:
                raise ValueError(f"Verification failed: missing key {key} in output file")
        
        # Calculate statistics
        pos_samples = int(np.sum(labels))
        neg_samples = n_samples - pos_samples
        file_size_mb = os.path.getsize(output_path) / (1024*1024)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"CONVERSION COMPLETED: {epigenome_id}-{marker}")
        print(f"{'='*50}")
        print(f"Output file: {output_filename}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"Total samples: {n_samples:,}")
        print(f"Positive samples: {pos_samples:,} ({pos_samples/n_samples*100:.1f}%)")
        print(f"Negative samples: {neg_samples:,} ({neg_samples/n_samples*100:.1f}%)")
        print(f"Target marker: {marker} (index {marker_idx})")
        
        # Verify one-hot encoding
        sample_sums = np.sum(dna_onehot[:5], axis=2)  # Check first 5 samples
        print(f"One-hot verification: {sample_sums.min():.1f} to {sample_sums.max():.1f} (should be 1.0)")
        
        if logger:
            logger.info(f"Successfully converted {epigenome_id}-{marker}: {n_samples:,} samples, {file_size_mb:.1f}MB")
        
        return output_path, True, "converted"
        
    except Exception as e:
        error_msg = f"Error converting {epigenome_id}-{marker}: {str(e)}"
        print(f"ERROR: {error_msg}")
        if logger:
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        if not config.CONTINUE_ON_ERROR:
            raise
        
        return None, False, "failed"


def convert_single_wrapper(args):
    """Wrapper function for multiprocessing"""
    epigenome_id, marker, input_path, input_filename = args
    return convert_single_dataset(epigenome_id, marker, input_path, input_filename)


def run_batch_conversion():
    """Run batch conversion of all pipeline outputs"""
    
    # Setup logging
    logger = setup_conversion_logging()
    logger.info("Starting batch conversion")
    
    # Find all pipeline outputs
    pipeline_files = find_pipeline_outputs()
    
    if not pipeline_files:
        print(f"No pipeline output files found in {config.INPUT_DIR}")
        print("Make sure your DeepHistone pipeline completed successfully.")
        return [], [], []
    
    total_files = len(pipeline_files)
    
    print(f"\n{'='*70}")
    print(f"BATCH DATASET CONVERSION")
    print(f"{'='*70}")
    print(f"Input directory: {config.INPUT_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Total files to convert: {total_files}")
    print(f"Multiprocessing: {config.USE_MULTIPROCESSING} ({config.N_PROCESSES} processes)")
    print(f"Skip existing: {config.SKIP_EXISTING}")
    print(f"Continue on error: {config.CONTINUE_ON_ERROR}")
    print(f"{'='*70}")
    
    # Show files to be processed
    print(f"\nFiles to process:")
    for epigenome_id, marker, input_path, filename in pipeline_files[:10]:  # Show first 10
        print(f"  {epigenome_id}-{marker}: {filename}")
    if total_files > 10:
        print(f"  ... and {total_files - 10} more files")
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    overall_start = time.time()
    
    if config.USE_MULTIPROCESSING and config.N_PROCESSES > 1:
        print(f"\nUsing multiprocessing with {config.N_PROCESSES} processes...")
        
        # Process in batches
        for batch_start in range(0, total_files, config.BATCH_SIZE):
            batch_end = min(batch_start + config.BATCH_SIZE, total_files)
            batch_files = pipeline_files[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//config.BATCH_SIZE + 1}: "
                  f"files {batch_start+1}-{batch_end}")
            
            with mp.Pool(processes=min(config.N_PROCESSES, len(batch_files))) as pool:
                results = pool.map(convert_single_wrapper, batch_files)
            
            # Collect results
            for (epigenome_id, marker, input_path, filename), (output_path, success, status) in zip(batch_files, results):
                if success:
                    if status == "skipped":
                        skipped.append((epigenome_id, marker, filename))
                    else:
                        successful.append((epigenome_id, marker, output_path))
                else:
                    failed.append((epigenome_id, marker, filename))
            
            # Print batch progress
            print(f"Batch completed. Current totals:")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"  Skipped: {len(skipped)}")
    
    else:
        print("\nUsing sequential processing...")
        
        for i, (epigenome_id, marker, input_path, filename) in enumerate(pipeline_files, 1):
            print(f"\n--- Converting {i}/{total_files}: {epigenome_id}-{marker} ---")
            
            try:
                output_path, success, status = convert_single_dataset(
                    epigenome_id, marker, input_path, filename, logger
                )
                
                if success:
                    if status == "skipped":
                        skipped.append((epigenome_id, marker, filename))
                    else:
                        successful.append((epigenome_id, marker, output_path))
                else:
                    failed.append((epigenome_id, marker, filename))
                    
            except Exception as e:
                error_msg = f"Unexpected error for {epigenome_id}-{marker}: {str(e)}"
                print(f"ERROR: {error_msg}")
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                failed.append((epigenome_id, marker, filename))
                
                if not config.CONTINUE_ON_ERROR:
                    raise
            
            # Progress update
            if i % 5 == 0 or i == total_files:
                elapsed = time.time() - overall_start
                avg_time = elapsed / i
                remaining = (total_files - i) * avg_time
                print(f"\nProgress: {i}/{total_files} ({i/total_files*100:.1f}%)")
                print(f"Elapsed: {elapsed/60:.1f}m, Estimated remaining: {remaining/60:.1f}m")
                print(f"Successful: {len(successful)}, Failed: {len(failed)}, Skipped: {len(skipped)}")
    
    # Final summary
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"BATCH CONVERSION COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total files: {total_files}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    
    if successful:
        print(f"\nSuccessfully converted:")
        total_size = 0
        for epigenome_id, marker, output_path in successful:
            file_size = os.path.getsize(output_path) / (1024*1024)
            total_size += file_size
            print(f"  {epigenome_id}-{marker}: {file_size:.1f}MB")
        print(f"Total converted data: {total_size:.1f}MB")
    
    if failed:
        print(f"\nFailed conversions:")
        for epigenome_id, marker, filename in failed:
            print(f"  {epigenome_id}-{marker}: {filename}")
    
    if skipped:
        print(f"\nSkipped (already existed):")
        for epigenome_id, marker, filename in skipped:
            print(f"  {epigenome_id}-{marker}: {filename}")
    
    logger.info(f"Batch conversion completed: {len(successful)} successful, {len(failed)} failed, {len(skipped)} skipped")
    
    return successful, failed, skipped


def quick_verify_converted_dataset(output_path):
    """Quick verification of a converted dataset"""
    
    print(f"\nQUICK VERIFICATION: {os.path.basename(output_path)}")
    print("="*50)
    
    if not os.path.exists(output_path):
        print("File not found")
        return False
    
    try:
        data = np.load(output_path)
        
        # Check expected format
        expected_shapes = {
            'keys': 'string identifiers',
            'dna': '(N, 1, 4, 1000) one-hot DNA',  
            'dnase': '(N, 1, 1, 1000) accessibility',
            'label': '(N, 1, 7) multi-task labels'
        }
        
        print("Format verification:")
        all_good = True
        for key, description in expected_shapes.items():
            if key in data.files:
                shape = data[key].shape
                dtype = data[key].dtype
                print(f"  {key:6}: {shape} {dtype} - {description}")
                
                # Basic validation
                if key == 'dna' and (len(shape) != 4 or shape[1:] != (1, 4, 1000)):
                    print(f"    WARNING: Unexpected DNA shape")
                    all_good = False
                elif key == 'dnase' and (len(shape) != 4 or shape[1:] != (1, 1, 1000)):
                    print(f"    WARNING: Unexpected DNase shape")
                    all_good = False
                elif key == 'label' and (len(shape) != 3 or shape[1:] != (1, 7)):
                    print(f"    WARNING: Unexpected label shape")
                    all_good = False
            else:
                print(f"  {key:6}: MISSING")
                all_good = False
        
        # Sample data check
        if all_good and 'keys' in data.files:
            n_samples = len(data['keys'])
            print(f"\nSample data (N={n_samples:,}):")
            print(f"  keys[0]: {data['keys'][0]}")
            
            # Check one-hot encoding
            dna_sum = np.sum(data['dna'][0, 0], axis=0)
            print(f"  DNA one-hot check: {dna_sum[:5]} (should be 1.0)")
            
            # Check DNase range
            dnase_range = f"{data['dnase'][0].min():.3f} to {data['dnase'][0].max():.3f}"
            print(f"  DNase range: {dnase_range}")
            
            # Check labels
            active_markers = np.where(data['label'][0, 0] > 0)[0]
            print(f"  Active markers: {active_markers} (indices in {config.ALL_MARKERS})")
        
        return all_good
        
    except Exception as e:
        print(f"Error during verification: {e}")
        return False


def main():
    """Main function for batch conversion"""
    
    print("DeepHistone Dataset Batch Converter")
    print("="*50)
    
    # Run batch conversion
    successful, failed, skipped = run_batch_conversion()
    
    # Verify a few converted files
    if successful:
        print(f"\nVerifying converted files...")
        verification_count = min(3, len(successful))
        
        for i in range(verification_count):
            epigenome_id, marker, output_path = successful[i]
            print(f"\nVerifying {i+1}/{verification_count}: {epigenome_id}-{marker}")
            is_valid = quick_verify_converted_dataset(output_path)
            if not is_valid:
                print(f"WARNING: Verification failed for {epigenome_id}-{marker}")
    
    print(f"\nConversion complete!")
    print(f"Check the '{config.OUTPUT_DIR}' directory for your converted datasets.")


if __name__ == "__main__":
    main()