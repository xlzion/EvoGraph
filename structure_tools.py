# structure_tools.py
# Functions for protein structure prediction using ESMFold

import torch
# Remove the direct import of esm, as we will use transformers
# import esm 
import logging
import numpy as np
# Add the import for the Hugging Face transformers model
from transformers import AutoTokenizer, EsmForProteinFolding

class StructureTools:
    def __init__(self, device):
        self.device = device
        logging.info("Initializing Structure Tools (ESMFold via Transformers)...")
        # The model will be loaded on demand to save memory
        self.model = None
        self.tokenizer = None # Add tokenizer for the folding model
        logging.info("Structure Tools initialized.")

    def _load_model(self):
        """Loads the ESMFold model using the transformers library."""
        if self.model is None:
            logging.info("Loading ESMFold model into memory via transformers...")
            # Use EsmForProteinFolding from the transformers library
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
            self.model = self.model.to(self.device)
            # Optional: Performance optimizations for modern GPUs
            if self.device.type == 'cuda':
                self.model.esm = self.model.esm.half()
            torch.backends.cuda.matmul.allow_tf32 = True
            # Optional: Chunk size for memory management on longer sequences
            self.model.trunk.set_chunk_size(64)
            
            self.model.eval()
            logging.info("ESMFold model loaded successfully.")

    def predict_structure(self, sequence):
        """Predicts the 3D structure of a protein sequence using ESMFold."""
        self._load_model()
        
        with torch.no_grad():
            # The transformers model has a built-in infer_pdb method
            output = self.model.infer_pdb(sequence)
        
        # The output is a PDB string
        pdb_string = output
        
        # Extract pLDDT from the PDB string (it's in the B-factor column)
        plddt_scores = []
        for line in pdb_string.split('\n'):
            if line.startswith("ATOM") and " CA " in line:
                try:
                    plddt = float(line[60:66].strip())
                    plddt_scores.append(plddt)
                except (ValueError, IndexError):
                    continue
        
        avg_plddt = np.mean(plddt_scores) if plddt_scores else 0.0
        
        return pdb_string, avg_plddt

    def calculate_f_struct(self, avg_plddt):
        """The structural fitness is simply the average pLDDT score."""
        return avg_plddt