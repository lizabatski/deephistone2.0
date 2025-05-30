import os
import numpy as np
from pyfaidx import Fasta
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial


class DeepHistoneConfig:
    def __init__(self):
        # parameters from paper
        self.WINDOW_SIZE = 200  # scanning windows
        self.FINAL_WINDOW_SIZE = 1000  # final sequences for model
        self.STEP_SIZE = 200    # non-overlapping scan 
        self.MIN_OVERLAP = 100  # minimum overlap with peak 
        self.MIN_SITES_THRESHOLD = 50000  # discard epigenomes with <50K sites
        self.RANDOM_SEED = 42
        
        #  7 histone markers from paper Table 1
        self.ALL_MARKERS = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K9ac']
        
        # The 15 epigenomes that passed their filtering (from Table 1)
        self.VALID_EPIGENOMES = [
            'E003', 'E004', 'E005', 'E006', 'E007', 'E011', 'E012', 
            'E013', 'E016', 'E024', 'E065', 'E066', 'E116', 'E117', 'E118'
        ]
        
        # paths
        self.BASE_PATH = "raw"
        self.CHROM_SIZES = "raw/hg19.chrom.sizes.txt"
        self.FASTA_PATH = "raw/hg19.fa"
        self.OUTPUT_DIR = "data"
        
        
        self.USE_MULTIPROCESSING = True
        self.N_PROCESSES = min(6, mp.cpu_count())
        
        
        self.MAX_N_FRACTION = 0.1
        self.VALIDATE_GENOME_COVERAGE = True
        
       
        self.TEST_MODE = True  # Set to True for chr22 only
        self.TEST_CHROMOSOME = "chr22"
        
    def get_chipseq_path(self, epigenome_id, marker):
        return f"{self.BASE_PATH}/{epigenome_id}-{marker}.narrowPeak"
    
    def get_dnase_path(self, epigenome_id):
        return f"{self.BASE_PATH}/{epigenome_id}-DNase.macs2.narrowPeak"
    
    def get_output_path(self, epigenome_id, target_marker):
        suffix = f"_{self.TEST_CHROMOSOME}" if self.TEST_MODE else ""
        return f"{self.OUTPUT_DIR}/{epigenome_id}_{target_marker}_deephistone{suffix}.npz"


config = DeepHistoneConfig()


def log_progress(message, start_time=None):
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"[{elapsed:.2f}s] {message}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
    return current_time


def load_chromosome_sizes():
    chrom_sizes = {}
    try:
        with open(config.CHROM_SIZES, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    chrom, size = parts[0], int(parts[1])
                    # test mode
                    if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                        continue
                    chrom_sizes[chrom] = size
    except FileNotFoundError:
        raise FileNotFoundError(f"Chromosome sizes file not found: {config.CHROM_SIZES}")
    
    if not chrom_sizes:
        raise ValueError("No chromosomes loaded - check file format and test mode settings")
    
    return chrom_sizes


def validate_epigenome_files(epigenome_id):

    missing_files = []
    
    
    for marker in config.ALL_MARKERS:
        chip_file = config.get_chipseq_path(epigenome_id, marker)
        if not os.path.exists(chip_file):
            missing_files.append(f"{marker} ChIP-seq")
    
    
    dnase_file = config.get_dnase_path(epigenome_id)
    if not os.path.exists(dnase_file):
        missing_files.append("DNase-seq")
    
    return len(missing_files) == 0, missing_files


def scan_genome_for_modification_sites(epigenome_id, marker, apply_threshold=True):
 
    start_time = log_progress(f"Scanning {epigenome_id}-{marker} for modification sites...")
    
    peaks_file = config.get_chipseq_path(epigenome_id, marker)
    
    if not os.path.exists(peaks_file):
        log_progress(f"Error: {peaks_file} not found")
        return []
    
    # chromosome size
    chrom_sizes = load_chromosome_sizes()
    
    # load and parse peaks
    peaks_by_chrom = defaultdict(list)
    total_peaks = 0
    
    with open(peaks_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip() #remove whitespace
            if not line or line.startswith('#'): #skip comments
                continue
                
            cols = line.split('\t') #split by tabs
            if len(cols) < 3:
                continue
                
            try:
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                
               
                if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                    continue
                
                # validate coordinates
                if start >= end or start < 0:
                    continue
                
                # make sure chromosome exists
                if chrom not in chrom_sizes:
                    continue
                
                # is peak in chromosme bnounds
                if end > chrom_sizes[chrom]:
                    end = chrom_sizes[chrom]
                
                if start < chrom_sizes[chrom]:  
                    peaks_by_chrom[chrom].append((start, end))
                    total_peaks += 1
                    
            except (ValueError, IndexError) as e:
                log_progress(f"Warning: Invalid line {line_num} in {peaks_file}: {e}")
                continue
    
    
    for chrom in peaks_by_chrom: #sort peaks to efficienctly search
        peaks_by_chrom[chrom].sort()
    
    log_progress(f"Loaded {total_peaks:,} peaks from {len(peaks_by_chrom)} chromosomes")
    
    
    modification_sites = [] #list to store windows that overlap with peaks
    total_windows = 0 #total windows scanned
    
    for chrom in sorted(chrom_sizes.keys()):
        if chrom not in peaks_by_chrom:
            continue
            
        chrom_size = chrom_sizes[chrom]
        chrom_peaks = peaks_by_chrom[chrom]
        chrom_sites = 0
        
    
        for window_start in range(0, chrom_size - config.WINDOW_SIZE + 1, config.STEP_SIZE):
            window_end = window_start + config.WINDOW_SIZE
            total_windows += 1
            
            
            has_sufficient_overlap = False
            
            for peak_start, peak_end in chrom_peaks:
                
                if peak_end <= window_start:
                    continue
                
                if peak_start >= window_end:
                    break
                
                # overlap
                overlap_start = max(window_start, peak_start)
                overlap_end = min(window_end, peak_end)
                overlap_length = overlap_end - overlap_start
                
                # at least 100bp overlap
                if overlap_length >= config.MIN_OVERLAP:
                    has_sufficient_overlap = True
                    break
            
            if has_sufficient_overlap:
                modification_sites.append((chrom, window_start, window_end)) #save as modification site
                chrom_sites += 1
        
        if chrom_sites > 0:
            log_progress(f"  {chrom}: {chrom_sites:,} modification sites")
    
    sites_count = len(modification_sites) #total sites found across all chromosomes
    log_progress(f"Found {sites_count:,} modification sites from {total_windows:,} windows scanned", start_time)
    
    
    if apply_threshold and sites_count < config.MIN_SITES_THRESHOLD and not config.TEST_MODE:
        log_progress(f"WARNING: Only {sites_count:,} sites found for {epigenome_id}-{marker}. "
                    f"Paper discards epigenomes with <{config.MIN_SITES_THRESHOLD:,} sites per marker.")
        return []
    
    return modification_sites


def load_all_histone_markers_for_epigenome(epigenome_id, target_marker):
    
    start_time = log_progress(f"Processing all histone markers for {epigenome_id}...")
    
    # validate files exists
    files_valid, missing_files = validate_epigenome_files(epigenome_id)
    if not files_valid:
        raise FileNotFoundError(f"Missing files for {epigenome_id}: {missing_files}")
    
    all_marker_sites = {}
    marker_stats = {}
    
    # process each marker
    for marker in config.ALL_MARKERS:
        marker_sites = scan_genome_for_modification_sites(epigenome_id, marker, apply_threshold=True)
        
        if not marker_sites and not config.TEST_MODE:
            log_progress(f"ERROR: {epigenome_id}-{marker} has insufficient sites, skipping epigenome")
            return None, None, None #the paper discards epigenomes missing any marker
        
        marker_sites_set = set(marker_sites) #converting to a set for faster operations
        all_marker_sites[marker] = marker_sites_set #each array element is a set
        marker_stats[marker] = len(marker_sites_set)
        
        log_progress(f"  {marker}: {len(marker_sites_set):,} sites")
    
    # extract target and non-target sites
    if target_marker not in all_marker_sites:
        raise ValueError(f"Target marker {target_marker} not found in processed markers")
    
    target_sites = all_marker_sites[target_marker] #this returns a set of target sites
    
    # combine other markers
    other_markers_sites = set()
    for marker, sites in all_marker_sites.items():
        if marker != target_marker:
            other_markers_sites.update(sites)
    
    # DeepHistone negative strategy: other markers - target marker
    negative_sites = other_markers_sites - target_sites
    
    # Calculate total unique sites across all markers
    all_sites_union = set()
    for sites in all_marker_sites.values():
        all_sites_union.update(sites)
    
    log_progress(f"Site statistics for {epigenome_id}:", start_time)
    log_progress(f"  Target ({target_marker}): {len(target_sites):,} sites")
    log_progress(f"  Other markers combined: {len(other_markers_sites):,} sites")
    log_progress(f"  Negatives (other - target): {len(negative_sites):,} sites")
    log_progress(f"  Total unique sites: {len(all_sites_union):,} sites")
    
    return list(target_sites), list(negative_sites), all_marker_sites


def extract_dnase_openness_scores(epigenome_id, regions):
    
    start_time = log_progress(f"Extracting DNase openness scores for {len(regions):,} regions...")
    
    dnase_file = config.get_dnase_path(epigenome_id)
    
    
    dnase_peaks_by_chrom = defaultdict(list) #dictionary to store DNase peaks by chromosome
    
    if not os.path.exists(dnase_file):
        log_progress(f"Warning: DNase file {dnase_file} not found, using zero openness scores")
        # Return all zeros
        return [np.zeros(config.FINAL_WINDOW_SIZE, dtype=np.float32) for _ in regions]
    
    # Parse DNase peaks
    total_dnase_peaks = 0
    with open(dnase_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            cols = line.split('\t')
            if len(cols) < 7:  # narrowPeak format needs at least 7 columns
                continue
                
            try:
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                
                # Filter for test mode
                if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                    continue
                
                # Extract fold enrichment score (column 7 in narrowPeak, 0-indexed = cols[6])
                # Some files may have foldChange in column 8 (cols[7])
                fold_enrichment = 1.0  # default
                
                try:
                    if len(cols) > 6:
                        #use signal value
                        fold_enrichment = float(cols[6])
                    else: 
                        fold_enrichment = 1.0
                except(ValueError, IndexError):
                     fold_enrichment = 1.0 
                
                # Ensure positive scores
                fold_enrichment = max(0.0, fold_enrichment)
                
                # add this peak with its coordinates and fold enrichment to the chromosome's peak list
                dnase_peaks_by_chrom[chrom].append((start, end, fold_enrichment))
                #total peaks
                total_dnase_peaks += 1
                
            except (ValueError, IndexError) as e:
                continue
    
    # Sort peaks for efficient lookup
    for chrom in dnase_peaks_by_chrom:
        dnase_peaks_by_chrom[chrom].sort()
    
    log_progress(f"Loaded {total_dnase_peaks:,} DNase peaks")
    
    # Extract openness scores for each region
    openness_scores = []
    
    for region_idx, (chrom, region_start, region_end) in enumerate(tqdm(regions, desc="Extracting openness")):
        region_length = region_end - region_start
        
        # Initialize with zeros -- non-peak positions get score 0
        openness = np.zeros(region_length, dtype=np.float32)
        
        # Apply fold enrichment scores to positions within DNase peaks
        if chrom in dnase_peaks_by_chrom:
            for peak_start, peak_end, fold_enrichment in dnase_peaks_by_chrom[chrom]:
                # Check for overlap
                if peak_end <= region_start or peak_start >= region_end:
                    continue
                
                # Calculate overlap positions
                overlap_start = max(region_start, peak_start)
                overlap_end = min(region_end, peak_end)
                
                if overlap_start < overlap_end:
                    # Convert to region-relative indices
                    start_idx = overlap_start - region_start
                    end_idx = overlap_end - region_start
                    
                    # Assign fold enrichment to all positions in overlap
                    openness[start_idx:end_idx] = fold_enrichment
        
        openness_scores.append(openness)
    
    log_progress(f"Extracted openness scores for {len(regions):,} regions", start_time)
    return openness_scores


def expand_regions_to_1000bp(regions_200bp):
    
    start_time = log_progress(f"Expanding {len(regions_200bp):,} regions from 200bp to 1000bp...")
    
    chrom_sizes = load_chromosome_sizes()
    expanded_regions = []
    filtered_count = 0
    
    #loop through each 200 bp region that was identified as a modification site
    for chrom, start_200, end_200 in regions_200bp:
        center = (start_200 + end_200) // 2
        
        # cCreate 1000bp window centered on this position
        half_final = config.FINAL_WINDOW_SIZE // 2
        start_1000 = center - half_final
        end_1000 = center + half_final
        
        # boundary checking
        if start_1000 < 0:
            start_1000 = 0
            end_1000 = config.FINAL_WINDOW_SIZE
        
        if chrom in chrom_sizes:
            chrom_size = chrom_sizes[chrom]
            if end_1000 > chrom_size:
                end_1000 = chrom_size
                start_1000 = max(0, chrom_size - config.FINAL_WINDOW_SIZE)
            
            # only keep if we can get a full 1000bp window
            if end_1000 - start_1000 >= config.FINAL_WINDOW_SIZE:

                end_1000 = start_1000 + config.FINAL_WINDOW_SIZE
                expanded_regions.append((chrom, start_1000, end_1000))
            else:
                filtered_count += 1
        else:
        
            if end_1000 - start_1000 >= config.FINAL_WINDOW_SIZE:
                expanded_regions.append((chrom, start_1000, start_1000 + config.FINAL_WINDOW_SIZE))
            else:
                filtered_count += 1
    
    if filtered_count > 0:
        log_progress(f"Filtered out {filtered_count:,} regions that couldn't form full 1000bp windows")
    
    log_progress(f"Successfully expanded {len(expanded_regions):,} regions", start_time)
    return expanded_regions


def extract_sequences(regions):
 
    start_time = log_progress(f"Extracting DNA sequences for {len(regions):,} regions...")
    
    try:
        genome = Fasta(config.FASTA_PATH)
    except Exception as e:
        raise FileNotFoundError(f"Cannot load genome FASTA: {config.FASTA_PATH}. Error: {e}")
    
    sequences = []
    invalid_count = 0
    
    for chrom, region_start, region_end in tqdm(regions, desc="Extracting sequences"):
        expected_length = region_end - region_start
        
        try:
            # Extract sequence
            seq = genome[chrom][region_start:region_end].seq.upper()
            
            # Validate length
            if len(seq) != expected_length:
                # Pad or truncate to expected length
                if len(seq) < expected_length:
                    seq = seq.ljust(expected_length, 'N')
                else:
                    seq = seq[:expected_length]
            
            # Quality check: count N's
            n_count = seq.count('N')
            n_fraction = n_count / len(seq)
            
            if n_fraction > config.MAX_N_FRACTION:
                invalid_count += 1
                # Replace with all N's to mark as low quality
                seq = 'N' * expected_length
            
            sequences.append(seq)
            
        except Exception as e:
            log_progress(f"Warning: Could not extract sequence for {chrom}:{region_start}-{region_end}: {e}")
            sequences.append('N' * expected_length)
            invalid_count += 1
    
    if invalid_count > 0:
        log_progress(f"Warning: {invalid_count:,} sequences had quality issues (>10% N's or extraction errors)")
    
    log_progress(f"Extracted {len(sequences):,} sequences", start_time)
    return sequences


def create_natural_imbalanced_dataset(pos_sequences, pos_openness, neg_sequences, neg_openness):
    
    start_time = log_progress("Creating dataset with natural class distribution...")
    
    pos_count = len(pos_sequences)
    neg_count = len(neg_sequences)
    
    if pos_count == 0:
        raise ValueError("No positive samples available")
    if neg_count == 0:
        raise ValueError("No negative samples available")
    
    # combine positive and negative sequences
    all_sequences = pos_sequences + neg_sequences
    # combine positive and negative opennes scores
    all_openness = pos_openness + neg_openness
    all_labels = np.array([1] * pos_count + [0] * neg_count, dtype=np.int32)
    
    
    natural_ratio = neg_count / pos_count
    log_progress(f"Created dataset: {pos_count:,} pos + {neg_count:,} neg = {len(all_sequences):,} total", start_time)
    log_progress(f"Natural class distribution ratio: {natural_ratio:.1f}:1 (negative:positive)")
    log_progress("Using natural imbalanced distribution as per DeepHistone paper methodology")
    
    return all_sequences, all_openness, all_labels


def save_dataset_with_metadata(output_path, sequences, openness, labels, epigenome_id, target_marker, 
                             genomic_keys=None,  
                             metadata=None):
    
    start_time = log_progress("Saving dataset...")
    
    # C output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert sequences to character array
    max_len = max(len(seq) for seq in sequences) if sequences else 0
    sequences_array = np.array([list(seq.ljust(max_len, 'N')) for seq in sequences], dtype='U1')
    openness_array = np.array(openness, dtype=np.float32)
    

    if genomic_keys:
        keys_array = np.array(genomic_keys, dtype='U30')
        print(f"Using provided genomic keys: {len(keys_array)}")
        print(f"Sample keys: {keys_array[:3]}")
    else:
        n_samples = len(sequences)
        chr22_start = 16000000
        keys = [f"chr22:{chr22_start + i*1200}-{chr22_start + i*1200 + 1000}" for i in range(n_samples)]
        keys_array = np.array(keys, dtype='U30')
        print(f"Created estimated genomic keys: {len(keys_array)}")
    
    # Prepare metadata
    save_metadata = {
        'epigenome_id': epigenome_id,
        'target_marker': target_marker,
        'window_size': config.WINDOW_SIZE,
        'final_window_size': config.FINAL_WINDOW_SIZE,
        'step_size': config.STEP_SIZE,
        'min_overlap': config.MIN_OVERLAP,
        'min_sites_threshold': config.MIN_SITES_THRESHOLD,
        'all_markers': config.ALL_MARKERS,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': config.TEST_MODE
    }
    
    if metadata:
        save_metadata.update(metadata)
    
    # Save compressed dataset WITH KEYS
    np.savez_compressed(
        output_path,
        sequences=sequences_array,
        openness=openness_array,
        labels=labels,
        keys=keys_array,  
        metadata=json.dumps(save_metadata)
    )
    
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\n{'='*50}")
    print(f"DATASET SAVED: {os.path.basename(output_path)}")
    print(f"{'='*50}")
    print(f"Epigenome: {epigenome_id}")
    print(f"Target marker: {target_marker}")
    print(f"Total samples: {len(sequences):,}")
    print(f"Sequence length: {len(sequences[0]) if sequences else 0} bp")
    print(f"Positive samples: {pos_count:,} ({pos_count/len(sequences)*100:.1f}%)")
    print(f"Negative samples: {neg_count:,} ({neg_count/len(sequences)*100:.1f}%)")
    if pos_count > 0 and neg_count > 0:
        print(f"Class ratio (neg:pos): {neg_count/pos_count:.1f}:1")
    print(f"Genomic keys: {len(keys_array):,} regions")
    print(f"Sample keys: {keys_array[:2].tolist()}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"File path: {output_path}")
    
    log_progress(f"Dataset saved successfully", start_time)
    return output_path


def run_deephistone_pipeline(epigenome_id, target_marker):
   
    overall_start = time.time()
    
    print(f"\n{'='*60}")
    print(f"DEEPHISTONE PIPELINE - PAPER-ACCURATE IMPLEMENTATION")
    print(f"{'='*60}")
    print(f"Epigenome: {epigenome_id}")
    print(f"Target marker: {target_marker}")
    print(f"Class distribution: Natural (imbalanced) as per paper methodology")
    print(f"Test mode: {config.TEST_MODE}")
    if config.TEST_MODE:
        print(f"Test chromosome: {config.TEST_CHROMOSOME}")
    print(f"{'='*60}")
    
    # Validate inputs
    if epigenome_id not in config.VALID_EPIGENOMES and not config.TEST_MODE:
        log_progress(f"Warning: {epigenome_id} not in validated epigenomes list from paper")
    
    if target_marker not in config.ALL_MARKERS:
        raise ValueError(f"Target marker {target_marker} not in valid markers: {config.ALL_MARKERS}")
    
    # Step 1: Load all histone markers for this epigenome
    target_sites_200bp, negative_sites_200bp, all_marker_sites = load_all_histone_markers_for_epigenome(
        epigenome_id, target_marker
    )
    
    if target_sites_200bp is None:
        raise ValueError(f"Failed to process epigenome {epigenome_id} - insufficient sites for some markers")
    
    # Step 2: Expand to 1000bp windows
    target_sites_1000bp = expand_regions_to_1000bp(target_sites_200bp)
    negative_sites_1000bp = expand_regions_to_1000bp(negative_sites_200bp)
    
    if len(target_sites_1000bp) == 0:
        raise ValueError("No valid positive sites after expansion to 1000bp")
    if len(negative_sites_1000bp) == 0:
        raise ValueError("No valid negative sites after expansion to 1000bp")
    
    # Step 3: Extract sequences
    pos_sequences = extract_sequences(target_sites_1000bp)
    neg_sequences = extract_sequences(negative_sites_1000bp)
    
    # Step 4: Extract DNase openness scores
    pos_openness = extract_dnase_openness_scores(epigenome_id, target_sites_1000bp)
    neg_openness = extract_dnase_openness_scores(epigenome_id, negative_sites_1000bp)
    
    # Step 5: Create dataset with natural class distribution (as per paper)
    sequences, openness, labels = create_natural_imbalanced_dataset(
        pos_sequences, pos_openness, neg_sequences, neg_openness
    )


    all_regions_1000bp = target_sites_1000bp + negative_sites_1000bp
    pos_labels = [1] * len(pos_sequences)
    neg_labels = [0] * len(neg_sequences)
    all_labels_before_shuffle = pos_labels + neg_labels
    
    genomic_keys = [f"{chrom}:{start}-{end}" for chrom, start, end in all_regions_1000bp]
    

    np.random.seed(config.RANDOM_SEED)
    indices = np.random.permutation(len(sequences))
    
   
    sequences = [sequences[i] for i in indices]
    openness = [openness[i] for i in indices]
    labels = labels[indices]
    genomic_keys = [genomic_keys[i] for i in indices]  

    metadata = {
        'uses_natural_distribution': True,
        'paper_methodology': 'DeepHistone - no artificial balancing',
        'original_positive_count': len(pos_sequences),
        'original_negative_count': len(neg_sequences),
        'final_dataset_size': len(sequences),
        'natural_ratio': len(neg_sequences) / len(pos_sequences) if len(pos_sequences) > 0 else 0
    }

    output_path = config.get_output_path(epigenome_id, target_marker)
    
   
    save_dataset_with_metadata(
        output_path, sequences, openness, labels, 
        epigenome_id, target_marker, 
        genomic_keys=genomic_keys, 
        metadata=metadata
    )
    

   
    
    total_time = time.time() - overall_start
    log_progress(f"=== Pipeline completed in {total_time:.2f} seconds ===")
    
    return output_path