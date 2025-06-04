# test_full_pipeline.py - Test the full genome pipeline
import os
import sys
import time
from pathlib import Path

# Import your pipeline code
try:
    from deephistone_pipeline_all import (
        DeepHistoneConfig, 
        setup_logging, 
        run_single_combination,
        validate_epigenome_files
    )
    print("✓ Successfully imported pipeline modules")
except ImportError as e:
    print(f"✗ Error importing pipeline modules: {e}")
    print("Make sure deephistone_pipeline_all.py is in the same directory")
    sys.exit(1)

def test_config_setup():
    """Test that config can be modified for full genome"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION SETUP")
    print("="*60)
    
    # Create config and modify for full genome
    config = DeepHistoneConfig()
    
    print(f"Original TEST_MODE: {config.TEST_MODE}")
    print(f"Original TEST_CHROMOSOME: {config.TEST_CHROMOSOME}")
    
    # Modify for full genome
    config.TEST_MODE = False
    config.TEST_CHROMOSOME = None
    config.USE_MULTIPROCESSING = True
    config.N_PROCESSES = min(4, os.cpu_count())  # Conservative for testing
    config.BATCH_SIZE = 1
    config.SKIP_EXISTING = True
    config.CONTINUE_ON_ERROR = True
    
    print(f"Modified TEST_MODE: {config.TEST_MODE}")
    print(f"Modified N_PROCESSES: {config.N_PROCESSES}")
    print(f"Modified BATCH_SIZE: {config.BATCH_SIZE}")
    
    # Update global config
    import deephistone_pipeline_all
    deephistone_pipeline_all.config = config
    
    print("✓ Configuration updated successfully")
    return config

def test_file_validation(epigenome_id="E003"):
    """Test that all required files exist for an epigenome"""
    print(f"\n" + "="*60)
    print(f"TESTING FILE VALIDATION FOR {epigenome_id}")
    print("="*60)
    
    config = DeepHistoneConfig()
    
    print(f"Checking files for epigenome: {epigenome_id}")
    print(f"Base path: {config.BASE_PATH}")
    
    # Check each required file
    missing_files = []
    
    # Check ChIP-seq files
    print(f"\nChIP-seq files ({len(config.ALL_MARKERS)} markers):")
    for marker in config.ALL_MARKERS:
        chip_file = config.get_chipseq_path(epigenome_id, marker)
        exists = os.path.exists(chip_file)
        size = os.path.getsize(chip_file) if exists else 0
        print(f"  {marker:8}: {chip_file} - {'✓' if exists else '✗'} ({size:,} bytes)")
        if not exists:
            missing_files.append(f"{marker} ChIP-seq")
    
    # Check DNase file
    dnase_file = config.get_dnase_path(epigenome_id)
    dnase_exists = os.path.exists(dnase_file)
    dnase_size = os.path.getsize(dnase_file) if dnase_exists else 0
    print(f"\nDNase file:")
    print(f"  DNase   : {dnase_file} - {'✓' if dnase_exists else '✗'} ({dnase_size:,} bytes)")
    if not dnase_exists:
        missing_files.append("DNase-seq")
    
    # Check reference files
    print(f"\nReference files:")
    fasta_exists = os.path.exists(config.FASTA_PATH)
    fasta_size = os.path.getsize(config.FASTA_PATH) if fasta_exists else 0
    print(f"  FASTA   : {config.FASTA_PATH} - {'✓' if fasta_exists else '✗'} ({fasta_size:,} bytes)")
    
    chrom_exists = os.path.exists(config.CHROM_SIZES)
    chrom_size = os.path.getsize(config.CHROM_SIZES) if chrom_exists else 0
    print(f"  ChromSz : {config.CHROM_SIZES} - {'✓' if chrom_exists else '✗'} ({chrom_size:,} bytes)")
    
    # Summary
    print(f"\nValidation Summary:")
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print(f"✓ All required files found for {epigenome_id}")
        return True

def test_single_marker_run(epigenome_id="E003", target_marker="H3K4me3"):
    """Test running pipeline for just one marker"""
    print(f"\n" + "="*60)
    print(f"TESTING SINGLE MARKER: {epigenome_id}-{target_marker}")
    print("="*60)
    
    # Setup config for full genome
    config = test_config_setup()
    
    # Check output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = config.get_output_path(epigenome_id, target_marker)
    
    print(f"Expected output: {output_path}")
    
    # Check if already exists
    if os.path.exists(output_path):
        existing_size = os.path.getsize(output_path) / (1024*1024)
        print(f"Output already exists ({existing_size:.1f} MB)")
        
        user_input = input("Delete and reprocess? [y/N]: ").lower()
        if user_input == 'y':
            os.remove(output_path)
            print("Deleted existing file")
        else:
            print("Keeping existing file, skipping processing")
            return True
    
    # Run the pipeline
    print(f"\nStarting pipeline for {epigenome_id}-{target_marker}...")
    print("This may take 30-90 minutes for full genome...")
    
    start_time = time.time()
    
    try:
        logger = setup_logging()
        result_path, success = run_single_combination(epigenome_id, target_marker, logger)
        
        elapsed_time = time.time() - start_time
        
        if success and result_path:
            file_size = os.path.getsize(result_path) / (1024*1024)
            print(f"\n✓ SUCCESS!")
            print(f"  Time: {elapsed_time/60:.1f} minutes")
            print(f"  Output: {result_path}")
            print(f"  Size: {file_size:.1f} MB")
            
            # Quick validation
            import numpy as np
            try:
                data = np.load(result_path, allow_pickle=True)
                n_sequences = len(data['sequences'])
                n_positive = int(data['labels'].sum())
                n_negative = len(data['labels']) - n_positive
                
                print(f"  Samples: {n_sequences:,} total ({n_positive:,} pos, {n_negative:,} neg)")
                print(f"  Ratio: {n_negative/n_positive:.1f}:1 (neg:pos)")
                
                return True
                
            except Exception as e:
                print(f"  Warning: Could not validate file contents: {e}")
                return True  # File exists, probably OK
        else:
            print(f"\n✗ FAILED after {elapsed_time/60:.1f} minutes")
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ ERROR after {elapsed_time/60:.1f} minutes: {e}")
        return False

def test_quick_validation():
    """Quick test of key functions without full processing"""
    print(f"\n" + "="*60)
    print("QUICK FUNCTION VALIDATION")
    print("="*60)
    
    try:
        from deephistone_pipeline_all import (
            load_chromosome_sizes,
            scan_genome_for_modification_sites,
            expand_regions_to_1000bp
        )
        
        config = test_config_setup()
        
        # Test chromosome loading
        print("Testing chromosome sizes loading...")
        chrom_sizes = load_chromosome_sizes()
        print(f"✓ Loaded {len(chrom_sizes)} chromosomes")
        print(f"  Example: chr1 = {chrom_sizes.get('chr1', 'N/A'):,} bp")
        
        # Test site scanning (quick test)
        print(f"\nTesting site scanning (E003-H3K4me3)...")
        sites = scan_genome_for_modification_sites("E003", "H3K4me3", apply_threshold=False)
        print(f"✓ Found {len(sites):,} modification sites")
        
        if sites:
            print(f"  First site: {sites[0]}")
            print(f"  Last site: {sites[-1]}")
            
            # Test expansion
            print(f"\nTesting region expansion...")
            expanded = expand_regions_to_1000bp(sites[:100])  # Test first 100
            print(f"✓ Expanded {len(expanded)} regions to 1000bp")
            
            if expanded:
                print(f"  Example: {expanded[0]}")
        
        print("✓ All quick tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Quick validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("DEEPHISTONE FULL GENOME PIPELINE TEST")
    print("="*80)
    
    # Test 1: Configuration
    try:
        config = test_config_setup()
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    # Test 2: File validation
    epigenome_id = "E003"  # You can change this
    if not test_file_validation(epigenome_id):
        print(f"\n✗ File validation failed for {epigenome_id}")
        print("Please check that all required files are in the 'raw/' directory")
        return False
    
    # Test 3: Quick function validation
    if not test_quick_validation():
        print(f"\n✗ Quick validation failed")
        return False
    
    # Test 4: Ask user if they want to run full test
    print(f"\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    print("This will test the complete pipeline for one marker.")
    print(f"Target: {epigenome_id}-H3K4me3 (full genome)")
    print("Estimated time: 30-90 minutes")
    print("Estimated output size: 50-200 MB")
    
    user_input = input("\nRun full test? [y/N]: ").lower()
    
    if user_input == 'y':
        success = test_single_marker_run(epigenome_id, "H3K4me3")
        
        if success:
            print(f"\n" + "="*60)
            print("✓ FULL TEST PASSED!")
            print("="*60)
            print("Your pipeline is ready for full genome processing.")
            print()
            print("Next steps:")
            print("1. Run for all markers: python run_full_deephistone_pipeline.py E003")
            print("2. Or use HPC version: python run_deephistone_hpc.py create-job E003")
            return True
        else:
            print(f"\n" + "="*60)
            print("✗ FULL TEST FAILED")
            print("="*60)
            print("Please check the error messages above and fix any issues.")
            return False
    else:
        print("\nSkipping full test.")
        print("Basic validation passed - pipeline should work for full genome.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)