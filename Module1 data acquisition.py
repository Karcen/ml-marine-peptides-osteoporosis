"""
Module 1: Data Acquisition
Downloads marine peptide data from NCBI Protein Database and CMNPD
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import Entrez, SeqIO
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# Import configuration
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NCBI_SEARCH_TERMS, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH,
    CMNPD_URL, DATA_DIRS, OUTPUT_DIRS
)


class NCBIDataDownloader:
    """Download peptide sequences from NCBI Protein database"""

    def __init__(self, email="researcher@example.com"):
        Entrez.email = email
        self.sequences = []
        self.metadata = []

    def search_ncbi(self, term, retmax=5000):
        """Search NCBI Protein database"""
        print(f"Searching NCBI for: {term}")

        try:
            handle = Entrez.esearch(
                db="protein",
                term=term,
                retmax=retmax,
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()

            count = int(record["Count"])
            print(f"  Found {count} records")

            return record
        except Exception as e:
            print(f"  Error searching: {e}")
            return None

    def fetch_sequences(self, record, batch_size=500):
        """Fetch sequences from search results"""
        if record is None:
            return []

        webenv = record["WebEnv"]
        query_key = record["QueryKey"]
        count = int(record["Count"])

        sequences = []

        for start in range(0, min(count, 5000), batch_size):
            print(f"  Fetching records {start + 1} to {min(start + batch_size, count)}...")

            try:
                fetch_handle = Entrez.efetch(
                    db="protein",
                    rettype="fasta",
                    retmode="text",
                    retstart=start,
                    retmax=batch_size,
                    webenv=webenv,
                    query_key=query_key
                )

                data = fetch_handle.read()
                fetch_handle.close()

                # Parse FASTA
                for record in SeqIO.parse(StringIO(data), "fasta"):
                    seq_len = len(record.seq)
                    if MIN_SEQ_LENGTH <= seq_len <= MAX_SEQ_LENGTH:
                        sequences.append({
                            "id": record.id,
                            "description": record.description,
                            "sequence": str(record.seq),
                            "length": seq_len
                        })

                time.sleep(0.5)  # Be nice to NCBI servers

            except Exception as e:
                print(f"  Error fetching: {e}")
                continue

        return sequences

    def download_all(self, output_path=None):
        """Download sequences for all search terms"""
        all_sequences = []

        for term in NCBI_SEARCH_TERMS:
            record = self.search_ncbi(term)
            sequences = self.fetch_sequences(record)
            all_sequences.extend(sequences)
            print(f"  Total sequences so far: {len(all_sequences)}")
            time.sleep(1)

        # Convert to DataFrame
        df = pd.DataFrame(all_sequences)

        if output_path is None:
            output_path = DATA_DIRS["ncbi"] / "ncbi_peptides_raw.csv"

        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} sequences to {output_path}")

        return df


class CMNPDDownloader:
    """Download and process CMNPD marine natural products database"""

    def __init__(self):
        self.data = None

    def download_cmnpd(self, output_path=None):
        """Download CMNPD database"""
        print("Downloading CMNPD database...")

        try:
            # Try direct download
            response = requests.get(CMNPD_URL, timeout=60)

            if response.status_code == 200:
                if output_path is None:
                    output_path = DATA_DIRS["cmnpd"] / "cmnpd_raw.tsv"

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                print(f"Downloaded CMNPD data to {output_path}")

                # Load and process
                self.data = pd.read_csv(output_path, sep='\t', low_memory=False)
                print(f"Loaded {len(self.data)} compounds")

                return self.data
            else:
                print(f"Download failed with status code: {response.status_code}")
                return self._download_pubchem_marine()

        except Exception as e:
            print(f"Error downloading CMNPD: {e}")
            print("Trying alternative source from PubChem...")
            return self._download_pubchem_marine()

    def _download_pubchem_marine(self, output_path=None):
        """Download marine compounds from PubChem as alternative"""
        print("Fetching marine natural products from PubChem...")

        try:
            # Alternative: search for known marine compound classes
            compounds_data = []

            marine_terms = [
                "marine peptide", "fish peptide", "algae peptide",
                "collagen peptide", "antimicrobial peptide marine"
            ]

            for term in marine_terms[:2]:  # Limit for speed
                search_response = requests.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{term.replace(' ', '%20')}/property/MolecularFormula,MolecularWeight,XLogP,TPSA,RotatableBondCount,HBondDonorCount,HBondAcceptorCount,IUPACName/CSV",
                    timeout=30
                )
                if search_response.status_code == 200:
                    df_temp = pd.read_csv(StringIO(search_response.text))
                    compounds_data.append(df_temp)
                time.sleep(1)

            if compounds_data:
                self.data = pd.concat(compounds_data, ignore_index=True)
                self.data = self.data.drop_duplicates(subset=['CID'])

                if output_path is None:
                    output_path = DATA_DIRS["cmnpd"] / "pubchem_marine.csv"

                self.data.to_csv(output_path, index=False)
                print(f"Saved {len(self.data)} compounds to {output_path}")

                return self.data
            else:
                self._print_data_unavailable_message()
                return None

        except Exception as e:
            print(f"PubChem download failed: {e}")
            self._print_data_unavailable_message()
            return None

    def _print_data_unavailable_message(self):
        """Print message when data is unavailable"""
        print("\n" + "=" * 60)
        print("ERROR: Unable to download marine compound data")
        print("=" * 60)
        print("\nPlease try one of the following options:")
        print("1. Check your internet connection and try again")
        print("2. Manually download CMNPD data from: https://www.cmnpd.org/")
        print("3. Download marine compounds from PubChem: https://pubchem.ncbi.nlm.nih.gov/")
        print("4. Use your own marine compound dataset in CSV format")
        print("\nRequired columns: compound_id, name, mw, logp, tpsa, hbd, hba")
        print("=" * 60 + "\n")


class PDBDownloader:
    """Download protein structures from RCSB PDB"""

    def __init__(self):
        self.structures = {}

    def download_pdb(self, pdb_id, output_dir=None):
        """Download PDB structure"""
        if output_dir is None:
            output_dir = DATA_DIRS["pdb"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_path = output_dir / f"{pdb_id}.pdb"

        print(f"Downloading PDB {pdb_id}...")

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"  Saved to {output_path}")
                self.structures[pdb_id] = output_path
                return output_path
            else:
                print(f"  Failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"  Error: {e}")
            return None

    def download_docking_targets(self):
        """Download all docking target structures"""
        from config import DOCKING_TARGETS

        results = {}
        for target_name, target_info in DOCKING_TARGETS.items():
            pdb_id = target_info["pdb_id"]
            path = self.download_pdb(pdb_id)
            results[target_name] = path

        return results


def download_all_data():
    """Main function to download all required data"""
    print("=" * 60)
    print("MARINE PEPTIDE DATA ACQUISITION")
    print("=" * 60)

    # Create directories
    for dir_path in DATA_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Download NCBI peptide sequences
    print("\n" + "=" * 40)
    print("1. Downloading NCBI Peptide Sequences")
    print("=" * 40)
    ncbi_downloader = NCBIDataDownloader()
    results['ncbi'] = ncbi_downloader.download_all()

    # 2. Download CMNPD database
    print("\n" + "=" * 40)
    print("2. Downloading CMNPD Database")
    print("=" * 40)
    cmnpd_downloader = CMNPDDownloader()
    results['cmnpd'] = cmnpd_downloader.download_cmnpd()

    # 3. Download PDB structures
    print("\n" + "=" * 40)
    print("3. Downloading PDB Structures")
    print("=" * 40)
    pdb_downloader = PDBDownloader()
    results['pdb'] = pdb_downloader.download_docking_targets()

    # Summary
    print("\n" + "=" * 60)
    print("DATA ACQUISITION SUMMARY")
    print("=" * 60)
    if results['ncbi'] is not None:
        print(f"NCBI Peptides: {len(results['ncbi'])} sequences")
    else:
        print("NCBI Peptides: Download failed - please check connection")
    if results['cmnpd'] is not None:
        print(f"CMNPD Compounds: {len(results['cmnpd'])} compounds")
    else:
        print("CMNPD Compounds: Download failed - please provide data manually")
    print(f"PDB Structures: {len(results['pdb'])} structures")

    return results


if __name__ == "__main__":
    download_all_data()