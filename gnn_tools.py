# gnn_tools.py
# Functions for GNN-based fitness prediction and crossover guidance

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import community as community_louvain
import numpy as np
import logging
import torch_geometric.data as geo_data
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer

from utils import parse_pdb_string_to_structure
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1, aa3, aa1

# Create a mapping from 1-letter code to an index for one-hot encoding
aa_map = {letter: i for i, letter in enumerate(aa1)}

# A placeholder GNN architecture similar to what might be used.
# In a real scenario, this would be a well-defined, pre-trained model.
class GNN_Oracle(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = gnn.global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class GNNTools:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.cutoff = config['models']['gnn']['graph_cutoff']
        
        logging.info("Initializing GNN Tools...")
        # Placeholder for node features dimension
        # (e.g., 20 for one-hot AA type + extra features)
        self.node_feature_dim = 21 
        
        try:
            self.oracle_model = GNN_Oracle(self.node_feature_dim, 128, 1).to(device)
            self.oracle_model.load_state_dict(torch.load(config['models']['gnn']['oracle_path']))
            self.oracle_model.eval()
            logging.info("GNN oracle model loaded successfully.")
        except FileNotFoundError:
            logging.warning("GNN oracle model not found. Using a dummy model. F_gnn will be random.")
            self.oracle_model = None
        except Exception as e:
            logging.error(f"Error loading GNN oracle: {e}. Using a dummy model.")
            self.oracle_model = None

    def build_protein_graph(self, pdb_string, sequence):
        """Converts a PDB structure string into a PyTorch Geometric graph."""
        structure = parse_pdb_string_to_structure(pdb_string)
        model = structure[0]  # Get the first model
        
        residues = [res for res in model.get_residues() if res.get_resname() in aa3]
        if not residues:
            logging.warning("Could not find any standard residues in the PDB string.")
            return None

        # Create node features (one-hot encoding of amino acid type)
        node_features = []
        for res in residues:
            res_name = res.get_resname()
            one_letter_code = protein_letters_3to1.get(res_name, 'X')
            
            one_hot = [0] * len(aa1)
            if one_letter_code in aa_map:
                one_hot[aa_map[one_letter_code]] = 1
            
            node_features.append(one_hot)
        
        x = torch.tensor(node_features, dtype=torch.float)

        # Create edge index from C-alpha distances
        edge_list = []
        coords = np.array([res["CA"].get_coord() for res in residues])
        dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
        
        adj = dist_matrix < self.cutoff
        np.fill_diagonal(adj, False) # No self-loops
        edge_index = torch.tensor(np.array(np.where(adj)), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, sequence=sequence)

    def calculate_f_gnn(self, graph_batch):
        """Calculates functional fitness (F_gnn) using the GNN oracle."""
        if not self.oracle_model:
            # Return random scores if no model is loaded
            return np.random.rand(len(graph_batch))
        
        loader = DataLoader(graph_batch, batch_size=len(graph_batch))
        batch = next(iter(loader)).to(self.device)
        
        with torch.no_grad():
            predictions = self.oracle_model(batch)
        
        return predictions.cpu().numpy().flatten()

    def get_structural_domains(self, graph):
        """Partitions a protein graph into structural domains using the Louvain algorithm."""
        if graph is None or graph.edge_index.shape == 0:
            return {i: 0 for i in range(graph.x.shape)} # Single domain if no edges

        # Convert to NetworkX graph for Louvain algorithm
        edge_list = graph.edge_index.t().tolist()
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(graph.num_nodes))
        nx_graph.add_edges_from(edge_list)
        
        # Apply Louvain community detection
        partition = community_louvain.best_partition(nx_graph)
        
        num_domains = len(set(partition.values()))
        logging.info(f"Identified {num_domains} structural domains.")
        
        return partition