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
import logging
from datetime import datetime
import traceback


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
        
        # Batch processing settings
        self.USE_MULTIPROCESSING = True
        self.N_PROCESSES = min(6, mp.cpu_count())
        self.BATCH_SIZE = 5  # Number of datasets to process in parallel
        
        self.MAX_N_FRACTION = 0.1
        self.VALIDATE_GENOME_COVERAGE = True
        
        # Test mode settings
        self.TEST_MODE = False  # Set to True for chr22 only
        self.TEST_CHROMOSOME = "chr22"
        
        # Batch processing options
        self.SKIP_EXISTING = True  # Skip already processed datasets
        self.CONTINUE_ON_ERROR = True  # Continue processing other datasets if one fails
        self.SAVE_PROGRESS = True  # Save progress periodically
        
    def get_chipseq_path(self, epigenome_id, marker):
        return f"{self.BASE_PATH}/{epigenome_id}-{marker}.narrowPeak"
    
    def get_dnase_path(self, epigenome_id):
        return f"{self.BASE_PATH}/{epigenome_id}-DNase.macs2.narrowPeak"
    
    def get_output_path(self, epigenome_id, target_marker):
        suffix = f"_{self.TEST_CHROMOSOME}" if self.TEST_MODE else ""
        return f"{self.OUTPUT_DIR}/{epigenome_id}_{target_marker}_deephistone{suffix}.npz"
    
    def get_all_combinations(self):
        """Generate all epigenome-marker combinations to process"""
        combinations = []
        for epigenome in self.VALID_EPIGENOMES:
            for marker in self.ALL_MARKERS:
                combinations.append((epigenome, marker))
        return combinations


config = DeepHistoneConfig()


def setup_logging():
    """Set up logging for batch processing"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/deephistone_batch_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


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
                    
                    if chrom in ['chrX', 'chrY']:
                        continue
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
    """Validate that all required files exist for an epigenome"""
    missing_files = []
    
    # Check ChIP-seq files for all markers
    for marker in config.ALL_MARKERS:
        chip_file = config.get_chipseq_path(epigenome_id, marker)
        if not os.path.exists(chip_file):
            missing_files.append(f"{marker} ChIP-seq")
    
    # Check DNase-seq file
    dnase_file = config.get_dnase_path(epigenome_id)
    if not os.path.exists(dnase_file):
        missing_files.append("DNase-seq")
    
    return len(missing_files) == 0, missing_files


def check_dataset_exists(epigenome_id, target_marker):
    """Check if dataset already exists"""
    output_path = config.get_output_path(epigenome_id, target_marker)
    return os.path.exists(output_path)


def validate_dataset_integrity(output_path):
    """Validate that a saved dataset is complete and not corrupted"""
    try:
        data = np.load(output_path, allow_pickle=True)
        required_keys = ['sequences', 'openness', 'labels', 'keys', 'metadata']
        
        for key in required_keys:
            if key not in data:
                return False, f"Missing key: {key}"
        
        # Check data consistency
        n_samples = len(data['sequences'])
        if (len(data['openness']) != n_samples or 
            len(data['labels']) != n_samples or 
            len(data['keys']) != n_samples):
            return False, "Inconsistent sample counts across arrays"
        
        # Check metadata
        try:
            metadata = json.loads(str(data['metadata']))
            if 'epigenome_id' not in metadata or 'target_marker' not in metadata:
                return False, "Missing required metadata"
        except:
            return False, "Invalid metadata format"
        
        return True, "Dataset is valid"
        
    except Exception as e:
        return False, f"Error loading dataset: {str(e)}"


def scan_genome_for_modification_sites(epigenome_id, marker, apply_threshold=True):
    """Scan genome for modification sites - same as original but with better error handling"""
    start_time = log_progress(f"Scanning {epigenome_id}-{marker} for modification sites...")
    
    peaks_file = config.get_chipseq_path(epigenome_id, marker)
    
    if not os.path.exists(peaks_file):
        log_progress(f"Error: {peaks_file} not found")
        return []
    
    # Load chromosome sizes
    chrom_sizes = load_chromosome_sizes()
    
    # Load and parse peaks
    peaks_by_chrom = defaultdict(list)
    total_peaks = 0
    
    with open(peaks_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            cols = line.split('\t')
            if len(cols) < 3:
                continue
                
            try:
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])

                if chrom in ['chrX', 'chrY']:
                   continue
                
                if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                    continue
                
                # Validate coordinates
                if start >= end or start < 0:
                    continue
                
                # Make sure chromosome exists
                if chrom not in chrom_sizes:
                    continue
                
                # Check peak is within chromosome bounds
                if end > chrom_sizes[chrom]:
                    end = chrom_sizes[chrom]
                
                if start < chrom_sizes[chrom]:  
                    peaks_by_chrom[chrom].append((start, end))
                    total_peaks += 1
                    
            except (ValueError, IndexError) as e:
                continue
    
    # Sort peaks for efficient searching
    for chrom in peaks_by_chrom:
        peaks_by_chrom[chrom].sort()
    
    log_progress(f"Loaded {total_peaks:,} peaks from {len(peaks_by_chrom)} chromosomes")
    
    # Scan genome with sliding windows
    modification_sites = []
    total_windows = 0
    
    for chrom in sorted(chrom_sizes.keys()):
        if chrom not in peaks_by_chrom:
            continue
            
        chrom_size = chrom_sizes[chrom]
        chrom_peaks = peaks_by_chrom[chrom]
        chrom_sites = 0
        
        for window_start in range(0, chrom_size - config.WINDOW_SIZE + 1, config.STEP_SIZE):
            window_end = window_start + config.WINDOW_SIZE
            total_windows += 1
            
            # Check for sufficient overlap with any peak
            has_sufficient_overlap = False
            
            for peak_start, peak_end in chrom_peaks:
                if peak_end <= window_start:
                    continue
                if peak_start >= window_end:
                    break
                
                # Calculate overlap
                overlap_start = max(window_start, peak_start)
                overlap_end = min(window_end, peak_end)
                overlap_length = overlap_end - overlap_start
                
                if overlap_length >= config.MIN_OVERLAP:
                    has_sufficient_overlap = True
                    break
            
            if has_sufficient_overlap:
                modification_sites.append((chrom, window_start, window_end))
                chrom_sites += 1
        
        if chrom_sites > 0:
            log_progress(f"  {chrom}: {chrom_sites:,} modification sites")
    
    sites_count = len(modification_sites)
    log_progress(f"Found {sites_count:,} modification sites from {total_windows:,} windows scanned", start_time)
    
    # Apply threshold check
    if apply_threshold and sites_count < config.MIN_SITES_THRESHOLD and not config.TEST_MODE:
        log_progress(f"WARNING: Only {sites_count:,} sites found for {epigenome_id}-{marker}. "
                    f"Paper discards epigenomes with <{config.MIN_SITES_THRESHOLD:,} sites per marker.")
        return []
    
    return modification_sites


def load_all_histone_markers_for_epigenome(epigenome_id, target_marker):
    """Load all histone markers for an epigenome - same as original"""
    start_time = log_progress(f"Processing all histone markers for {epigenome_id}...")
    
    # Validate files exist
    files_valid, missing_files = validate_epigenome_files(epigenome_id)
    if not files_valid:
        raise FileNotFoundError(f"Missing files for {epigenome_id}: {missing_files}")
    
    all_marker_sites = {}
    marker_stats = {}
    
    # Process each marker
    for marker in config.ALL_MARKERS:
        marker_sites = scan_genome_for_modification_sites(epigenome_id, marker, apply_threshold=True)
        
        if not marker_sites and not config.TEST_MODE:
            log_progress(f"ERROR: {epigenome_id}-{marker} has insufficient sites, skipping epigenome")
            return None, None, None
        
        marker_sites_set = set(marker_sites)
        all_marker_sites[marker] = marker_sites_set
        marker_stats[marker] = len(marker_sites_set)
        
        log_progress(f"  {marker}: {len(marker_sites_set):,} sites")
    
    # Extract target and non-target sites
    if target_marker not in all_marker_sites:
        raise ValueError(f"Target marker {target_marker} not found in processed markers")
    
    target_sites = all_marker_sites[target_marker]
    
    # Combine other markers
    other_markers_sites = set()
    for marker, sites in all_marker_sites.items():
        if marker != target_marker:
            other_markers_sites.update(sites)
    
    # DeepHistone negative strategy: other markers - target marker
    negative_sites = other_markers_sites - target_sites
    
    # Calculate statistics
    all_sites_union = set()
    for sites in all_marker_sites.values():
        all_sites_union.update(sites)
    
    log_progress(f"Site statistics for {epigenome_id}:", start_time)
    log_progress(f"  Target ({target_marker}): {len(target_sites):,} sites")
    log_progress(f"  Other markers combined: {len(other_markers_sites):,} sites")
    log_progress(f"  Negatives (other - target): {len(negative_sites):,} sites")
    log_progress(f"  Total unique sites: {len(all_sites_union):,} sites")
    
    return list(target_sites), list(negative_sites), all_marker_sites


# [Include all other functions from original code: expand_regions_to_1000bp, extract_sequences, 
#  extract_dnase_openness_scores, create_natural_imbalanced_dataset, save_dataset_with_metadata]

def expand_regions_to_1000bp(regions_200bp):
    """Expand 200bp regions to 1000bp - same as original"""
    start_time = log_progress(f"Expanding {len(regions_200bp):,} regions from 200bp to 1000bp...")
    
    chrom_sizes = load_chromosome_sizes()
    expanded_regions = []
    filtered_count = 0
    
    for chrom, start_200, end_200 in regions_200bp:
        center = (start_200 + end_200) // 2
        
        # Create 1000bp window centered on this position
        half_final = config.FINAL_WINDOW_SIZE // 2
        start_1000 = center - half_final
        end_1000 = center + half_final
        
        # Boundary checking
        if start_1000 < 0:
            start_1000 = 0
            end_1000 = config.FINAL_WINDOW_SIZE
        
        if chrom in chrom_sizes:
            chrom_size = chrom_sizes[chrom]
            if end_1000 > chrom_size:
                end_1000 = chrom_size
                start_1000 = max(0, chrom_size - config.FINAL_WINDOW_SIZE)
            
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
    """Extract DNA sequences - same as original"""
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
            seq = genome[chrom][region_start:region_end].seq.upper()
            
            if len(seq) != expected_length:
                if len(seq) < expected_length:
                    seq = seq.ljust(expected_length, 'N')
                else:
                    seq = seq[:expected_length]
            
            n_count = seq.count('N')
            n_fraction = n_count / len(seq)
            
            if n_fraction > config.MAX_N_FRACTION:
                invalid_count += 1
                seq = 'N' * expected_length
            
            sequences.append(seq)
            
        except Exception as e:
            log_progress(f"Warning: Could not extract sequence for {chrom}:{region_start}-{region_end}: {e}")
            sequences.append('N' * expected_length)
            invalid_count += 1
    
    if invalid_count > 0:
        log_progress(f"Warning: {invalid_count:,} sequences had quality issues")
    
    log_progress(f"Extracted {len(sequences):,} sequences", start_time)
    return sequences


def extract_dnase_openness_scores(epigenome_id, regions):
    """Extract DNase openness scores - same as original"""
    start_time = log_progress(f"Extracting DNase openness scores for {len(regions):,} regions...")
    
    dnase_file = config.get_dnase_path(epigenome_id)
    dnase_peaks_by_chrom = defaultdict(list)
    
    if not os.path.exists(dnase_file):
        log_progress(f"Warning: DNase file {dnase_file} not found, using zero openness scores")
        return [np.zeros(config.FINAL_WINDOW_SIZE, dtype=np.float32) for _ in regions]
    
    # Parse DNase peaks
    total_dnase_peaks = 0
    with open(dnase_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            cols = line.split('\t')
            if len(cols) < 7:
                continue
                
            try:
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])

                if chrom in ['chrX', 'chrY']:
                    continue
                
                if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                    continue
                
                fold_enrichment = 1.0
                try:
                    if len(cols) > 6:
                        fold_enrichment = float(cols[6])
                    else: 
                        fold_enrichment = 1.0
                except(ValueError, IndexError):
                     fold_enrichment = 1.0 
                
                fold_enrichment = max(0.0, fold_enrichment)
                dnase_peaks_by_chrom[chrom].append((start, end, fold_enrichment))
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
        openness = np.zeros(region_length, dtype=np.float32)
        
        if chrom in dnase_peaks_by_chrom:
            for peak_start, peak_end, fold_enrichment in dnase_peaks_by_chrom[chrom]:
                if peak_end <= region_start or peak_start >= region_end:
                    continue
                
                overlap_start = max(region_start, peak_start)
                overlap_end = min(region_end, peak_end)
                
                if overlap_start < overlap_end:
                    start_idx = overlap_start - region_start
                    end_idx = overlap_end - region_start
                    openness[start_idx:end_idx] = fold_enrichment
        
        openness_scores.append(openness)
    
    log_progress(f"Extracted openness scores for {len(regions):,} regions", start_time)
    return openness_scores


def create_natural_imbalanced_dataset(pos_sequences, pos_openness, neg_sequences, neg_openness):
    """Create dataset with natural class distribution - same as original"""
    start_time = log_progress("Creating dataset with natural class distribution...")
    
    pos_count = len(pos_sequences)
    neg_count = len(neg_sequences)
    
    if pos_count == 0:
        raise ValueError("No positive samples available")
    if neg_count == 0:
        raise ValueError("No negative samples available")
    
    all_sequences = pos_sequences + neg_sequences
    all_openness = pos_openness + neg_openness
    all_labels = np.array([1] * pos_count + [0] * neg_count, dtype=np.int32)
    
    natural_ratio = neg_count / pos_count
    log_progress(f"Created dataset: {pos_count:,} pos + {neg_count:,} neg = {len(all_sequences):,} total", start_time)
    log_progress(f"Natural class distribution ratio: {natural_ratio:.1f}:1 (negative:positive)")
    
    return all_sequences, all_openness, all_labels


def save_dataset_with_metadata(output_path, sequences, openness, labels, epigenome_id, target_marker, 
                             genomic_keys=None, metadata=None):
    """Save dataset with metadata - same as original"""
    start_time = log_progress("Saving dataset...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    max_len = max(len(seq) for seq in sequences) if sequences else 0
    sequences_array = np.array([list(seq.ljust(max_len, 'N')) for seq in sequences], dtype='U1')
    openness_array = np.array(openness, dtype=np.float32)
    
    if genomic_keys:
        keys_array = np.array(genomic_keys, dtype='U30')
    else:
        n_samples = len(sequences)
        chr22_start = 16000000
        keys = [f"chr22:{chr22_start + i*1200}-{chr22_start + i*1200 + 1000}" for i in range(n_samples)]
        keys_array = np.array(keys, dtype='U30')
    
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
    print(f"Positive samples: {pos_count:,} ({pos_count/len(sequences)*100:.1f}%)")
    print(f"Negative samples: {neg_count:,} ({neg_count/len(sequences)*100:.1f}%)")
    if pos_count > 0 and neg_count > 0:
        print(f"Class ratio (neg:pos): {neg_count/pos_count:.1f}:1")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"File path: {output_path}")
    
    log_progress(f"Dataset saved successfully", start_time)
    return output_path


def run_single_combination(epigenome_id, target_marker, logger=None):
    """Run the DeepHistone pipeline for a single epigenome-marker combination"""
    overall_start = time.time()

    try:
        if logger:
            logger.info(f"Starting {epigenome_id}-{target_marker}")

        print(f"\n{'='*60}")
        print(f"Processing: {epigenome_id} - {target_marker}")
        print(f"{'='*60}")

        # Check if already exists and skip if requested
        if getattr(config, "SKIP_EXISTING", False) and check_dataset_exists(epigenome_id, target_marker):
            output_path = config.get_output_path(epigenome_id, target_marker)
            is_valid, msg = validate_dataset_integrity(output_path)
            if is_valid:
                print(f"Dataset already exists and is valid: {output_path}")
                if logger:
                    logger.info(f"Skipped {epigenome_id}-{target_marker} - already exists")
                return output_path, True
            else:
                print(f"Existing dataset is invalid ({msg}), reprocessing...")
                if logger:
                    logger.warning(f"Invalid dataset for {epigenome_id}-{target_marker}: {msg}")

        # Validate marker
        if target_marker not in config.ALL_MARKERS:
            raise ValueError(f"Target marker '{target_marker}' not in valid marker list: {config.ALL_MARKERS}")

        # Step 1: Load positive and negative sites
        target_sites_200bp, negative_sites_200bp, all_marker_sites = load_all_histone_markers_for_epigenome(
            epigenome_id, target_marker
        )

        if target_sites_200bp is None:
            raise ValueError(f"Failed to process {epigenome_id} - missing sufficient data for at least one marker")

        # Step 2: Expand 200bp to 1000bp regions
        target_sites_1000bp = expand_regions_to_1000bp(target_sites_200bp)
        negative_sites_1000bp = expand_regions_to_1000bp(negative_sites_200bp)

        if len(target_sites_1000bp) == 0:
            raise ValueError("No valid positive regions after expansion to 1000bp")
        if len(negative_sites_1000bp) == 0:
            raise ValueError("No valid negative regions after expansion to 1000bp")

        # Step 3: Extract DNA sequences
        pos_sequences = extract_sequences(target_sites_1000bp)
        neg_sequences = extract_sequences(negative_sites_1000bp)

        # Step 4: Extract DNase-seq openness scores
        pos_openness = extract_dnase_openness_scores(epigenome_id, target_sites_1000bp)
        neg_openness = extract_dnase_openness_scores(epigenome_id, negative_sites_1000bp)

        # Step 5: Create dataset with natural class imbalance
        sequences, openness, labels = create_natural_imbalanced_dataset(
            pos_sequences, pos_openness, neg_sequences, neg_openness
        )

        # Step 6: Shuffle dataset
        all_regions_1000bp = target_sites_1000bp + negative_sites_1000bp
        genomic_keys = [f"{chrom}:{start}-{end}" for chrom, start, end in all_regions_1000bp]

        np.random.seed(config.RANDOM_SEED)
        indices = np.random.permutation(len(sequences))

        sequences = [sequences[i] for i in indices]
        openness = [openness[i] for i in indices]
        labels = labels[indices]
        genomic_keys = [genomic_keys[i] for i in indices]

        # Step 7: Metadata for reproducibility
        metadata = {
            'uses_natural_distribution': True,
            'paper_methodology': 'DeepHistone',
            'original_positive_count': len(pos_sequences),
            'original_negative_count': len(neg_sequences),
            'final_dataset_size': len(sequences),
            'natural_ratio': len(neg_sequences) / len(pos_sequences) if len(pos_sequences) > 0 else 0
        }

        # Step 8: Save the dataset
        output_path = config.get_output_path(epigenome_id, target_marker)
        save_dataset_with_metadata(
            output_path, sequences, openness, labels,
            epigenome_id, target_marker,
            genomic_keys=genomic_keys,
            metadata=metadata
        )

        duration = time.time() - overall_start
        log_progress(f"Finished {epigenome_id}-{target_marker} in {duration:.2f} seconds")
        if logger:
            logger.info(f"Completed {epigenome_id}-{target_marker} in {duration:.2f} seconds")

        return output_path, True

    except Exception as e:
        error_msg = f"Error processing {epigenome_id}-{target_marker}: {e}"
        print(error_msg)
        if logger:
            logger.error(error_msg)
        return None, False
