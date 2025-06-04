import os
import sys
import time
from deephistone_pipeline_all import (
    config, 
    setup_logging, 
    run_single_combination
)

def run_single_epigenome(epigenome_id):
    """Run pipeline for all markers of a single epigenome"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING EPIGENOME: {epigenome_id}")
    print(f"{'='*80}")
    print(f"Test mode: {config.TEST_MODE}")
    print(f"Markers to process: {config.ALL_MARKERS}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"{'='*80}")
    
    logger = setup_logging()
    logger.info(f"Starting full processing for {epigenome_id}")
    
    successful = []
    failed = []
    overall_start = time.time()
    
    for i, marker in enumerate(config.ALL_MARKERS, 1):
        print(f"\n--- Processing {i}/{len(config.ALL_MARKERS)}: {epigenome_id}-{marker} ---")
        
        try:
            start_time = time.time()
            output_path, success = run_single_combination(epigenome_id, marker, logger)
            elapsed = time.time() - start_time
            
            if success:
                print(f"SUCCESS: {epigenome_id}-{marker} ({elapsed/60:.1f} minutes)")
                successful.append((epigenome_id, marker))
            else:
                print(f"FAILED: {epigenome_id}-{marker}")
                failed.append((epigenome_id, marker))
                
        except Exception as e:
            print(f"ERROR: {epigenome_id}-{marker}: {e}")
            failed.append((epigenome_id, marker))
            logger.error(f"Error processing {epigenome_id}-{marker}: {e}")
    
    # Final summary
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {epigenome_id}")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Successful: {len(successful)}/{len(config.ALL_MARKERS)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful datasets:")
        for epi, marker in successful:
            output_path = config.get_output_path(epi, marker)
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024*1024)
                print(f"  {epi}-{marker}: {file_size:.1f} MB")
    
    if failed:
        print(f"\nFailed datasets:")
        for epi, marker in failed:
            print(f"  {epi}-{marker}")
    
    logger.info(f"Completed {epigenome_id}: {len(successful)} successful, {len(failed)} failed")
    return successful, failed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_epigenome.py <epigenome_id>")
        print("Example: python run_epigenome.py E003")
        sys.exit(1)
    
    epigenome_id = sys.argv[1]
    
    
    if epigenome_id not in config.VALID_EPIGENOMES:
        print(f"Error: {epigenome_id} not in valid epigenomes: {config.VALID_EPIGENOMES}")
        sys.exit(1)
    
    print(f"Starting processing for epigenome: {epigenome_id}")
    run_single_epigenome(epigenome_id)