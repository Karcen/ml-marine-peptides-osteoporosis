"""
Module 3: Sequence Processing and Physicochemical Properties
Handles sequence deduplication, clustering, and property calculation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import hashlib
from Bio.Align import PairwiseAligner
import warnings

warnings.filterwarnings('ignore')

import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLUSTERING_THRESHOLD, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH,
    AMINO_ACID_MW, KYTE_DOOLITTLE, PKA_VALUES, HYDROPHOBIC_AA,
    DATA_DIRS, OUTPUT_DIRS
)


class SequenceProcessor:
    """Process and deduplicate peptide sequences"""

    def __init__(self):
        self.raw_sequences = None
        self.unique_sequences = None
        self.clustered_sequences = None
        self.cluster_map = {}

    def load_sequences(self, filepath):
        """Load sequences from CSV file"""
        print(f"Loading sequences from {filepath}...")
        self.raw_sequences = pd.read_csv(filepath)
        print(f"  Loaded {len(self.raw_sequences)} sequences")
        return self.raw_sequences

    def deduplicate_exact(self):
        """Remove exact duplicate sequences using hash table"""
        print("Performing exact deduplication...")

        seen_hashes = {}
        unique_indices = []

        for idx, row in self.raw_sequences.iterrows():
            seq = str(row['sequence']).upper()
            seq_hash = hashlib.md5(seq.encode()).hexdigest()

            if seq_hash not in seen_hashes:
                seen_hashes[seq_hash] = idx
                unique_indices.append(idx)

        self.unique_sequences = self.raw_sequences.loc[unique_indices].copy()
        self.unique_sequences.reset_index(drop=True, inplace=True)

        removed = len(self.raw_sequences) - len(self.unique_sequences)
        print(f"  Removed {removed} exact duplicates")
        print(f"  Remaining: {len(self.unique_sequences)} unique sequences")

        return self.unique_sequences

    def cluster_sequences(self, threshold=None):
        """Cluster sequences by similarity using global alignment"""
        if threshold is None:
            threshold = CLUSTERING_THRESHOLD

        print(f"Clustering sequences at {threshold * 100}% identity threshold...")

        if self.unique_sequences is None:
            self.deduplicate_exact()

        sequences = self.unique_sequences['sequence'].tolist()
        n_seqs = len(sequences)

        # Sort by length (descending) for efficiency
        sorted_indices = sorted(range(n_seqs),
                                key=lambda i: len(sequences[i]),
                                reverse=True)

        # Cluster representatives
        representatives = []
        cluster_assignments = {}

        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5

        for i, idx in enumerate(sorted_indices):
            if i % 100 == 0:
                print(f"  Processing sequence {i + 1}/{n_seqs}...")

            seq = sequences[idx]
            seq_len = len(seq)
            assigned = False

            for rep_idx in representatives:
                rep_seq = sequences[rep_idx]
                rep_len = len(rep_seq)

                # Length filter for efficiency
                len_diff = abs(seq_len - rep_len) / min(seq_len, rep_len)
                if len_diff > 0.1:
                    continue

                # Calculate alignment
                try:
                    alignments = aligner.align(seq, rep_seq)
                    if alignments:
                        score = alignments[0].score
                        max_len = max(seq_len, rep_len)
                        identity = score / max_len

                        if identity >= threshold:
                            cluster_assignments[idx] = rep_idx
                            assigned = True
                            break
                except:
                    continue

            if not assigned:
                representatives.append(idx)
                cluster_assignments[idx] = idx

        # Create clustered DataFrame
        self.clustered_sequences = self.unique_sequences.loc[representatives].copy()
        self.clustered_sequences.reset_index(drop=True, inplace=True)
        self.cluster_map = cluster_assignments

        print(f"  Created {len(representatives)} clusters from {n_seqs} sequences")

        return self.clustered_sequences

    def filter_by_length(self, min_len=None, max_len=None):
        """Filter sequences by length"""
        if min_len is None:
            min_len = MIN_SEQ_LENGTH
        if max_len is None:
            max_len = MAX_SEQ_LENGTH

        df = self.clustered_sequences if self.clustered_sequences is not None else self.unique_sequences

        mask = (df['length'] >= min_len) & (df['length'] <= max_len)
        filtered = df[mask].copy()
        filtered.reset_index(drop=True, inplace=True)

        print(f"Filtered to {len(filtered)} sequences (length {min_len}-{max_len})")

        return filtered


class PhysicochemicalCalculator:
    """Calculate physicochemical properties of peptides"""

    def __init__(self):
        self.properties = None

    def calculate_molecular_weight(self, sequence):
        """Calculate peptide molecular weight"""
        sequence = sequence.upper()

        # Sum amino acid weights
        mw = sum(AMINO_ACID_MW.get(aa, 0) for aa in sequence)

        # Subtract water for peptide bonds
        mw -= (len(sequence) - 1) * 18.015

        return mw

    def calculate_charge(self, sequence, ph=7.4):
        """Calculate net charge at given pH using Henderson-Hasselbalch"""
        sequence = sequence.upper()

        # Positive charges (protonated forms)
        charge = 0

        # N-terminus
        charge += 1 / (1 + 10 ** (ph - PKA_VALUES['N_term']))

        # Basic residues (K, R, H)
        for aa in sequence:
            if aa == 'K':
                charge += 1 / (1 + 10 ** (ph - PKA_VALUES['K']))
            elif aa == 'R':
                charge += 1 / (1 + 10 ** (ph - PKA_VALUES['R']))
            elif aa == 'H':
                charge += 1 / (1 + 10 ** (ph - PKA_VALUES['H']))

        # Negative charges (deprotonated forms)
        # C-terminus
        charge -= 1 / (1 + 10 ** (PKA_VALUES['C_term'] - ph))

        # Acidic residues (D, E, C, Y)
        for aa in sequence:
            if aa == 'D':
                charge -= 1 / (1 + 10 ** (PKA_VALUES['D'] - ph))
            elif aa == 'E':
                charge -= 1 / (1 + 10 ** (PKA_VALUES['E'] - ph))
            elif aa == 'C':
                charge -= 1 / (1 + 10 ** (PKA_VALUES['C'] - ph))
            elif aa == 'Y':
                charge -= 1 / (1 + 10 ** (PKA_VALUES['Y'] - ph))

        return charge

    def calculate_hydrophobicity(self, sequence):
        """Calculate average Kyte-Doolittle hydrophobicity"""
        sequence = sequence.upper()

        if len(sequence) == 0:
            return 0

        total = sum(KYTE_DOOLITTLE.get(aa, 0) for aa in sequence)
        return total / len(sequence)

    def calculate_hydrophobic_ratio(self, sequence):
        """Calculate ratio of hydrophobic residues"""
        sequence = sequence.upper()

        if len(sequence) == 0:
            return 0

        hydrophobic_count = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA)
        return hydrophobic_count / len(sequence)

    def calculate_isoelectric_point(self, sequence, precision=0.01):
        """Calculate isoelectric point (pI) by binary search"""
        sequence = sequence.upper()

        ph_low, ph_high = 0, 14

        while (ph_high - ph_low) > precision:
            ph_mid = (ph_low + ph_high) / 2
            charge = self.calculate_charge(sequence, ph_mid)

            if charge > 0:
                ph_low = ph_mid
            else:
                ph_high = ph_mid

        return (ph_low + ph_high) / 2

    def calculate_amino_acid_composition(self, sequence):
        """Calculate amino acid composition"""
        sequence = sequence.upper()
        composition = defaultdict(int)

        for aa in sequence:
            composition[aa] += 1

        # Convert to percentages
        total = len(sequence)
        if total > 0:
            for aa in composition:
                composition[aa] = composition[aa] / total * 100

        return dict(composition)

    def calculate_aromaticity(self, sequence):
        """Calculate aromaticity (fraction of aromatic amino acids)"""
        sequence = sequence.upper()
        aromatic = set(['F', 'W', 'Y'])

        if len(sequence) == 0:
            return 0

        return sum(1 for aa in sequence if aa in aromatic) / len(sequence)

    def calculate_instability_index(self, sequence):
        """Calculate instability index (simplified version)"""
        sequence = sequence.upper()

        # DIWV matrix values for common dipeptides (simplified)
        # A protein is considered unstable if index > 40
        DIWV = {
            'WW': 1.0, 'WC': 1.0, 'WM': 24.68,
            'CC': 1.0, 'FF': 1.0, 'MV': -7.49,
            'GM': -6.54, 'GY': -7.09, 'AN': 1.0,
            # Simplified - using average for most pairs
        }

        if len(sequence) < 2:
            return 0

        score = 0
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i + 2]
            score += DIWV.get(dipeptide, 1.0)

        return (10.0 / len(sequence)) * score

    def calculate_all_properties(self, sequences_df):
        """Calculate all properties for a DataFrame of sequences"""
        print("Calculating physicochemical properties...")

        properties = []

        for idx, row in sequences_df.iterrows():
            if idx % 100 == 0:
                print(f"  Processing sequence {idx + 1}/{len(sequences_df)}...")

            seq = str(row['sequence']).upper()

            # Filter non-standard amino acids
            seq = ''.join(aa for aa in seq if aa in AMINO_ACID_MW)

            if len(seq) < MIN_SEQ_LENGTH:
                continue

            prop = {
                'id': row.get('id', f'SEQ_{idx}'),
                'sequence': seq,
                'length': len(seq),
                'molecular_weight': self.calculate_molecular_weight(seq),
                'charge_ph7': self.calculate_charge(seq, 7.4),
                'hydrophobicity': self.calculate_hydrophobicity(seq),
                'hydrophobic_ratio': self.calculate_hydrophobic_ratio(seq),
                'isoelectric_point': self.calculate_isoelectric_point(seq),
                'aromaticity': self.calculate_aromaticity(seq),
                'instability_index': self.calculate_instability_index(seq),
            }

            # Add amino acid composition
            composition = self.calculate_amino_acid_composition(seq)
            for aa in AMINO_ACID_MW.keys():
                prop[f'aa_{aa}'] = composition.get(aa, 0)

            properties.append(prop)

        self.properties = pd.DataFrame(properties)
        print(f"  Calculated properties for {len(self.properties)} sequences")

        return self.properties

    def calculate_scoring_function(self, properties_df, reference_means=None, reference_stds=None):
        """Calculate multi-dimensional scoring function"""
        print("Calculating comprehensive scoring function...")

        # Features to use for scoring
        score_features = [
            'molecular_weight', 'hydrophobicity', 'charge_ph7',
            'hydrophobic_ratio', 'aromaticity'
        ]

        # Get reference statistics (if not provided, use dataset statistics)
        if reference_means is None:
            reference_means = properties_df[score_features].mean()
        if reference_stds is None:
            reference_stds = properties_df[score_features].std()

        # Calculate Gaussian-based scores for each feature
        scores = pd.DataFrame()

        for feature in score_features:
            mu = reference_means[feature]
            sigma = reference_stds[feature]
            if sigma > 0:
                scores[f'{feature}_score'] = np.exp(
                    -((properties_df[feature] - mu) ** 2) / (2 * sigma ** 2)
                )
            else:
                scores[f'{feature}_score'] = 1.0

        # Calculate composite score (geometric mean)
        score_cols = [f'{f}_score' for f in score_features]
        properties_df['composite_score'] = scores[score_cols].prod(axis=1) ** (1 / len(score_features))

        # 新增: 根据审稿意见2，添加长度惩罚项以纠正偏倚
        properties_df['composite_score'] *= np.exp(-properties_df['length'] / 100)  # 指数衰减惩罚长序列

        print(
            f"  Score range: {properties_df['composite_score'].min():.4f} - {properties_df['composite_score'].max():.4f}")

        return properties_df


def process_sequences(input_path=None, output_dir=None):
    """Main function to process sequences"""
    print("=" * 60)
    print("SEQUENCE PROCESSING AND PROPERTY CALCULATION")
    print("=" * 60)

    if output_dir is None:
        output_dir = OUTPUT_DIRS["chapter1_database"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    if input_path is None:
        input_path = DATA_DIRS["ncbi"] / "ncbi_peptides_raw.csv"

    processor = SequenceProcessor()

    if Path(input_path).exists():
        processor.load_sequences(input_path)
    else:
        print("\n" + "=" * 60)
        print("ERROR: Input sequence file not found!")
        print("=" * 60)
        print(f"\nExpected file: {input_path}")
        print("\nPlease run Module1_data_acquisition.py first to download")
        print("peptide sequences from NCBI, or provide your own sequence file.")
        print("\nRequired CSV format:")
        print("  - id: sequence identifier")
        print("  - sequence: amino acid sequence")
        print("  - length: sequence length")
        print("  - description: (optional) sequence description")
        print("\nExample:")
        print("  id,sequence,length,description")
        print("  PEPT_001,GPRPGPAG,8,Sample peptide")
        print("=" * 60 + "\n")
        return None

    # Process sequences
    processor.deduplicate_exact()
    clustered = processor.cluster_sequences()

    # Save clustered sequences
    clustered.to_csv(output_dir / "clustered_sequences.csv", index=False)

    # Calculate properties
    calculator = PhysicochemicalCalculator()
    properties = calculator.calculate_all_properties(clustered)
    properties = calculator.calculate_scoring_function(properties)

    # Save properties
    properties.to_csv(output_dir / "sequence_properties.csv", index=False)

    # Summary statistics
    summary = properties.describe()
    summary.to_csv(output_dir / "properties_summary.csv")

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Raw sequences: {len(processor.raw_sequences)}")
    print(f"After deduplication: {len(processor.unique_sequences)}")
    print(f"After clustering: {len(clustered)}")
    print(f"\nProperty statistics saved to: {output_dir / 'properties_summary.csv'}")

    return properties


if __name__ == "__main__":
    process_sequences()