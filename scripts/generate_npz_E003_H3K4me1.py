import os
import numpy as np
from pyfaidx import Fasta
import pyranges as pr

# Paths
CHROM_SIZES = "raw/hg19.chrom.sizes.txt"
FASTA_PATH = "raw/hg19.fa"
CHIPSEQ_PEAKS = "raw/E003-H3K4me1.narrowPeak"
DNASE_PEAKS = "raw/E003-DNase.macs2.narrowPeak"
OUTPUT_NPZ = "data/E003_H3K4me1_positive_only.npz"
WINDOW_SIZE = 200
EXPANDED_SIZE = 1000
OVERLAP_THRESHOLD = 100

# genreate windoes of size 200
def generate_windows(chrom_sizes_file, window=200):
    windows = []
    with open(chrom_sizes_file, 'r') as f:
        for line in f:
            chrom, size = line.strip().split()
            size = int(size)
            for start in range(0, size - window + 1, window):
                end = start + window
                windows.append([chrom, start, end])
    return pr.PyRanges(pd.DataFrame(windows, columns=["Chromosome", "Start", "End"]))

# find positive windows
def get_positive_windows(windows_gr, peaks_file):
    peaks = pr.read_bed(peaks_file)
    # minimum 100 bp overlap
    overlap = windows_gr.join(peaks, strandedness=False)
    overlap = overlap[(overlap.End_b - overlap.Start_b) >= OVERLAP_THRESHOLD]
    return overlap[["Chromosome", "Start", "End"]]

# expand to 1000 bp
def expand_to_1000bp(df):
    expanded = []
    for _, row in df.iterrows():
        center = (row["Start"] + row["End"]) // 2
        new_start = max(0, center - EXPANDED_SIZE // 2)
        new_end = new_start + EXPANDED_SIZE
        expanded.append([row["Chromosome"], new_start, new_end])
    return expanded

# extract dna sequence
def extract_sequences(fasta_path, regions):
    genome = Fasta(fasta_path)
    seqs = []
    for chrom, start, end in regions:
        try:
            seq = genome[chrom][start:end].seq.upper()
            seqs.append(seq)
        except:
            seqs.append("N" * (end - start))
    return seqs

# extract dna openess
def parse_dnase_signal(dnase_peak_file, regions):
    dnase_dict = {}
    with open(dnase_peak_file, 'r') as f:
        for line in f:
            cols = line.strip().split()
            chrom, start, end, *rest = cols
            score = float(cols[9])
            if chrom not in dnase_dict:
                dnase_dict[chrom] = []
            dnase_dict[chrom].append((int(start), int(end), score))

    openness = []
    for chrom, start, end in regions:
        signal = np.zeros(end - start)
        if chrom in dnase_dict:
            for peak_start, peak_end, score in dnase_dict[chrom]:
                overlap_start = max(start, peak_start)
                overlap_end = min(end, peak_end)
                if overlap_start < overlap_end:
                    signal[overlap_start - start:overlap_end - start] = score
        openness.append(signal)
    return openness

# Save to .npz
def save_npz(path, sequences, openness):
    np.savez_compressed(path, sequences=sequences, openness=openness, labels=np.ones(len(sequences)))
    print(f"Saved: {path}")


if __name__ == "__main__":
    import pandas as pd

    print("Generating genome windows...")
    windows_gr = generate_windows(CHROM_SIZES)

    print("Filtering positive windows...")
    positives_gr = get_positive_windows(windows_gr, CHIPSEQ_PEAKS)
    positives_df = positives_gr.df.reset_index(drop=True)
    print(f"Found {len(positives_df)} positive 200bp windows.")

    print("Expanding to 1000bp...")
    expanded = expand_to_1000bp(positives_df)

    print("Extracting DNA sequences...")
    sequences = extract_sequences(FASTA_PATH, expanded)

    print("Extracting DNase openness...")
    openness = parse_dnase_signal(DNASE_PEAKS, expanded)

    print("Saving to .npz...")
    save_npz(OUTPUT_NPZ, sequences, openness)
