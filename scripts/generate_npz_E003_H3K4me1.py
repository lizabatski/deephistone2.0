import os
import numpy as np
from pyfaidx import Fasta
import pyranges as pr
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm

# Paths
CHROM_SIZES = "raw/hg19.chrom.sizes.txt"
FASTA_PATH = "raw/hg19.fa"
CHIPSEQ_PEAKS = "raw/E003-H3K4me1.narrowPeak"
DNASE_PEAKS = "raw/E003-DNase.macs2.narrowPeak"
OUTPUT_NPZ = "data/E003_H3K4me1_positive_only.npz"
WINDOW_SIZE = 200
EXPANDED_SIZE = 1000
OVERLAP_THRESHOLD = 100

def log_progress(message, start_time=None):
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"[{elapsed:.2f}s] {message}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
    return current_time

# windows of size 200
def generate_windows(chrom_sizes_file, window=200):
    start_time = log_progress("Starting window generation...")
    windows = []
    with open(chrom_sizes_file, 'r') as f:
        for line in f:
            chrom, size = line.strip().split()
            size = int(size)
            for start in range(0, size - window + 1, window):
                end = start + window
                windows.append([chrom, start, end])
    
    log_progress(f"Generated {len(windows)} windows", start_time)
    return pr.PyRanges(pd.DataFrame(windows, columns=["Chromosome", "Start", "End"]))

# finding positive windows
def get_positive_windows(windows_gr, peaks_file):
    start_time = log_progress("Loading and filtering positive windows...")
    peaks = pr.read_bed(peaks_file)
    # minimum 100 bp overlap
    overlap = windows_gr.join(peaks, strandedness=False)
    overlap = overlap[(overlap.End_b - overlap.Start_b) >= OVERLAP_THRESHOLD]
    result = overlap[["Chromosome", "Start", "End"]]
    log_progress(f"Found {len(result)} positive windows", start_time)
    return result

# expand to 1000 bp
def expand_to_1000bp(df):
    start_time = log_progress("Expanding windows to 1000bp...")
    expanded = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Expanding windows"):
        center = (row["Start"] + row["End"]) // 2
        new_start = max(0, center - EXPANDED_SIZE // 2)
        new_end = new_start + EXPANDED_SIZE
        expanded.append([row["Chromosome"], new_start, new_end])
    
    log_progress(f"Expanded {len(expanded)} regions", start_time)
    return expanded

# extract dna sequence
def extract_sequences(fasta_path, regions):
    start_time = log_progress("Loading genome and extracting sequences...")
    genome = Fasta(fasta_path)
    seqs = []
    
    for chrom, start, end in tqdm(regions, desc="Extracting sequences"):
        try:
            seq = genome[chrom][start:end].seq.upper()
            seqs.append(seq)
        except:
            seqs.append("N" * (end - start))
    
    log_progress(f"Extracted {len(seqs)} sequences", start_time)
    return seqs

# was taking way to long- still does
def parse_dnase_signal_optimized(dnase_peak_file, regions):
    start_time = log_progress("Loading DNase peaks...")
    
    
    dnase_peaks_by_chrom = defaultdict(list)
    peak_count = 0
    
    with open(dnase_peak_file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 10:
                continue
            chrom, start, end = cols[0], int(cols[1]), int(cols[2])
            try:
                score = float(cols[9])  # signalValue column
            except (ValueError, IndexError):
                score = 1.0  # default score if parsing fails
            
            dnase_peaks_by_chrom[chrom].append((start, end, score))
            peak_count += 1
    
    log_progress(f"Loaded {peak_count} DNase peaks across {len(dnase_peaks_by_chrom)} chromosomes", start_time)
    
    # sort peaks by chromosome for faster lookup
    sort_start = time.time()
    for chrom in dnase_peaks_by_chrom:
        dnase_peaks_by_chrom[chrom].sort(key=lambda x: x[0])  
    log_progress(f"Sorted peaks for efficient lookup", sort_start)
    
    # Step 3: Process regions in batches by chromosome
    extract_start = time.time()
    openness = []
    
    # Group regions by chromosome for batch processing
    regions_by_chrom = defaultdict(list)
    for i, (chrom, start, end) in enumerate(regions):
        regions_by_chrom[chrom].append((i, start, end))
    
    # Initialize results array
    results = [None] * len(regions)
    
    # Process each chromosome
    for chrom in tqdm(regions_by_chrom.keys(), desc="Processing chromosomes"):
        chrom_regions = regions_by_chrom[chrom]
        chrom_peaks = dnase_peaks_by_chrom.get(chrom, [])
        
        if not chrom_peaks:
            # No peaks for this chromosome, fill with zeros
            for region_idx, start, end in chrom_regions:
                results[region_idx] = np.zeros(end - start)
            continue
        
        # Process regions for this chromosome
        for region_idx, region_start, region_end in chrom_regions:
            signal = np.zeros(region_end - region_start)
            
            # Binary search to find relevant peaks
            relevant_peaks = []
            for peak_start, peak_end, score in chrom_peaks:
                # Skip peaks that end before our region starts
                if peak_end <= region_start:
                    continue
                # Stop when peaks start after our region ends
                if peak_start >= region_end:
                    break
                # This peak overlaps with our region
                relevant_peaks.append((peak_start, peak_end, score))
            
            # Apply signals from relevant peaks
            for peak_start, peak_end, score in relevant_peaks:
                overlap_start = max(region_start, peak_start)
                overlap_end = min(region_end, peak_end)
                if overlap_start < overlap_end:
                    signal[overlap_start - region_start:overlap_end - region_start] = score
            
            results[region_idx] = signal
    
    # Convert results back to list in original order
    openness = [results[i] for i in range(len(results))]
    
    log_progress(f"Extracted DNase openness for {len(openness)} regions", extract_start)
    return openness

# Alternative even faster version using vectorized operations
def parse_dnase_signal_vectorized(dnase_peak_file, regions):
    """Ultra-fast version using PyRanges for overlap detection"""
    start_time = log_progress("Starting vectorized DNase signal extraction...")
    
    # Load DNase peaks into PyRanges
    load_start = time.time()
    dnase_df = []
    with open(dnase_peak_file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 10:
                continue
            chrom, start, end = cols[0], int(cols[1]), int(cols[2])
            try:
                score = float(cols[9])
            except (ValueError, IndexError):
                score = 1.0
            dnase_df.append([chrom, start, end, score])
    
    dnase_df = pd.DataFrame(dnase_df, columns=['Chromosome', 'Start', 'End', 'Score'])
    dnase_gr = pr.PyRanges(dnase_df)
    log_progress(f"Loaded {len(dnase_df)} DNase peaks into PyRanges", load_start)
    
    # Convert regions to PyRanges
    regions_df = pd.DataFrame(regions, columns=['Chromosome', 'Start', 'End'])
    regions_df['RegionID'] = range(len(regions_df))
    regions_gr = pr.PyRanges(regions_df)
    
    # Find overlaps
    overlap_start = time.time()
    overlaps = regions_gr.join(dnase_gr, strandedness=False)
    log_progress(f"Found overlaps using PyRanges", overlap_start)
    
    # Process overlaps to create signal arrays
    signal_start = time.time()
    openness = []
    
    for i, (chrom, region_start, region_end) in enumerate(tqdm(regions, desc="Creating signal arrays")):
        signal = np.zeros(region_end - region_start)
        
        # Get overlaps for this region
        region_overlaps = overlaps[overlaps.RegionID == i]
        
        if len(region_overlaps) > 0:
            for _, row in region_overlaps.df.iterrows():
                peak_start, peak_end, score = row['Start_b'], row['End_b'], row['Score']
                overlap_start_pos = max(region_start, peak_start)
                overlap_end_pos = min(region_end, peak_end)
                
                if overlap_start_pos < overlap_end_pos:
                    signal[overlap_start_pos - region_start:overlap_end_pos - region_start] = score
        
        openness.append(signal)
    
    log_progress(f"Created signal arrays for {len(openness)} regions", signal_start)
    return openness

# Save to .npz
def save_npz(path, sequences, openness):
    start_time = log_progress("Saving data to .npz file...")
    np.savez_compressed(path, sequences=sequences, openness=openness, labels=np.ones(len(sequences)))
    log_progress(f"Saved to: {path}", start_time)

if __name__ == "__main__":
    overall_start = time.time()
    log_progress("=== Starting genomic data processing pipeline ===")

    log_progress("Generating genome windows...")
    windows_gr = generate_windows(CHROM_SIZES)

    log_progress("Filtering positive windows...")
    positives_gr = get_positive_windows(windows_gr, CHIPSEQ_PEAKS)
    positives_df = positives_gr.df.reset_index(drop=True)
    log_progress(f"Found {len(positives_df)} positive 200bp windows.")

    log_progress("Expanding to 1000bp...")
    expanded = expand_to_1000bp(positives_df)

    log_progress("Extracting DNA sequences...")
    sequences = extract_sequences(FASTA_PATH, expanded)

    log_progress("Extracting DNase openness (optimized)...")
    # Use the optimized version - try vectorized first, fall back to optimized if needed
    try:
        openness = parse_dnase_signal_vectorized(DNASE_PEAKS, expanded)
    except Exception as e:
        log_progress(f"Vectorized method failed ({e}), falling back to optimized method...")
        openness = parse_dnase_signal_optimized(DNASE_PEAKS, expanded)

    log_progress("Saving to .npz...")
    save_npz(OUTPUT_NPZ, sequences, openness)

    total_time = time.time() - overall_start
    log_progress(f"=== Pipeline completed in {total_time:.2f} seconds ===")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total regions processed: {len(sequences)}")
    print(f"Sequence length: {len(sequences[0]) if sequences else 0} bp")
    print(f"Average DNase signal per region: {np.mean([np.mean(sig) for sig in openness]):.4f}")
    print(f"Output file: {OUTPUT_NPZ}")
    print(f"File size: {os.path.getsize(OUTPUT_NPZ) / (1024*1024):.1f} MB" if os.path.exists(OUTPUT_NPZ) else "File not found")