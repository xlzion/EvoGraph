# utils.py
# Utility functions for the EvoGraph-PLM framework
import logging
import os
import yaml
from Bio.PDB import PDBParser
from io import StringIO

def setup_logging(log_file):
    """Sets up logging to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_pdb_string(pdb_string, filepath):
    """Saves a PDB string to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(pdb_string)
    logging.info(f"Saved PDB file to {filepath}")

def parse_pdb_string_to_structure(pdb_string, structure_id="protein"):
    """Parses a PDB string into a BioPython Structure object."""
    parser = PDBParser(QUIET=True)
    handle = StringIO(pdb_string)
    structure = parser.get_structure(structure_id, handle)
    return structure