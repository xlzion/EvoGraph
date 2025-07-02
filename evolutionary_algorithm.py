# evolutionary_algorithm.py
# Main script for the EvoGraph-PLM optimization loop.

import os
import numpy as np
import logging
import random
import torch
from utils import load_config, setup_logging, save_pdb_string
from plm_tools import PLMTools
from structure_tools import StructureTools
from gnn_tools import GNNTools

class Individual:
    """A class to represent one individual in the population."""
    def __init__(self, sequence):
        self.sequence = sequence
        self.pdb_string = None
        self.graph = None
        self.domains = None
        
        self.f_seq = -np.inf
        self.f_struct = -np.inf
        self.f_gnn = -np.inf
        self.fitness = -np.inf

    def __repr__(self):
        return f"Individual(Seq='{self.sequence[:10]}...', Fitness={self.fitness:.4f})"

class EvoGraphPLM:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        setup_logging(config['output']['log_file'])
        logging.info(f"Running on device: {self.device}")
        
        self.plm = PLMTools(config, self.device)
        self.structure = StructureTools(self.device)
        self.gnn = GNNTools(config, self.device)
        
        self.population = []
        self.best_individual = None

    def evaluate_population(self):
        """Evaluates the fitness of the entire population."""
        logging.info("Evaluating population...")
        
        # Batch evaluate F_seq
        sequences = [ind.sequence for ind in self.population]
        f_seq_scores = self.plm.calculate_f_seq(sequences)

        graph_batch = []
        for i, ind in enumerate(self.population):
            ind.f_seq = f_seq_scores[i]
            
            # Predict structure and get F_struct
            pdb_string, avg_plddt = self.structure.predict_structure(ind.sequence)
            ind.pdb_string = pdb_string
            ind.f_struct = self.structure.calculate_f_struct(avg_plddt)
            
            # Build graph
            graph = self.gnn.build_protein_graph(pdb_string, ind.sequence)
            if graph:
                ind.graph = graph
                graph_batch.append(graph)
            else:
                # Penalize individuals that fail to produce a valid graph
                ind.f_gnn = -np.inf

        # Batch evaluate F_gnn
        if graph_batch:
            f_gnn_scores = self.gnn.calculate_f_gnn(graph_batch)
            graph_idx = 0
            for ind in self.population:
                if ind.graph is not None:
                    ind.f_gnn = f_gnn_scores[graph_idx]
                    graph_idx += 1

        # Calculate final weighted fitness
        w = self.config['fitness_weights']
        for ind in self.population:
            ind.fitness = (w['w_seq'] * ind.f_seq + 
                           w['w_struct'] * ind.f_struct + 
                           w['w_gnn'] * ind.f_gnn)
            logging.info(f"Individual fitness evaluation:")
            logging.info(f"  Sequence: {ind.sequence[:30]}{'...' if len(ind.sequence) > 30 else ''}")
            logging.info(f"  F_seq: {ind.f_seq:.3f}, F_struct: {ind.f_struct:.3f}, F_gnn: {ind.f_gnn:.3f}")
            logging.info(f"  Total fitness: {ind.fitness:.4f}")

    def selection(self):
        """Performs tournament selection to choose parents."""
        selected_parents = []
        for _ in range(self.config['ea']['population_size']):
            tournament = random.sample(self.population, self.config['ea']['tournament_size'])
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected_parents.append(winner)
        return selected_parents

    def crossover(self, parent1, parent2):
        """Performs GNN-guided crossover."""
        if parent1.graph is None or parent2.graph is None:
            return Individual(parent1.sequence), Individual(parent2.sequence)

        # Get domains for both parents
        parent1.domains = self.gnn.get_structural_domains(parent1.graph)
        parent2.domains = self.gnn.get_structural_domains(parent2.graph)

        # Find a domain to swap (simple strategy: choose a random domain from parent1)
        domain_ids1 = list(set(parent1.domains.values()))
        if not domain_ids1: return Individual(parent1.sequence), Individual(parent2.sequence)
        
        domain_to_swap1 = random.choice(domain_ids1)
        
        # Find a corresponding domain in parent2 (simple strategy: random)
        domain_ids2 = list(set(parent2.domains.values()))
        if not domain_ids2: return Individual(parent1.sequence), Individual(parent2.sequence)
        
        domain_to_swap2 = random.choice(domain_ids2)

        # Get residue indices for the selected domains
        indices1 = {i for i, dom_id in parent1.domains.items() if dom_id == domain_to_swap1}
        indices2 = {i for i, dom_id in parent2.domains.items() if dom_id == domain_to_swap2}

        if not indices1 or not indices2:
            return Individual(parent1.sequence), Individual(parent2.sequence)

        # Extract sequence segments from the original sequences
        seg1 = "".join([parent1.sequence[i] for i in sorted(list(indices1))])
        seg2 = "".join([parent2.sequence[i] for i in sorted(list(indices2))])
        
        # --- CORRECTED CROSSOVER LOGIC ---
        
        # Create Child 1: Take parent1's sequence, remove domain1, and insert domain2
        child1_list = [residue for i, residue in enumerate(parent1.sequence) if i not in indices1]
        insertion_point1 = min(indices1)
        child1_final_seq = "".join(child1_list[:insertion_point1]) + seg2 + "".join(child1_list[insertion_point1:])
        child1 = Individual(child1_final_seq)

        # Create Child 2: Take parent2's sequence, remove domain2, and insert domain1
        child2_list = [residue for i, residue in enumerate(parent2.sequence) if i not in indices2]
        insertion_point2 = min(indices2)
        child2_final_seq = "".join(child2_list[:insertion_point2]) + seg1 + "".join(child2_list[insertion_point2:])
        child2 = Individual(child2_final_seq)
        
        logging.info(f"Crossover performed, creating children of length {len(child1.sequence)} and {len(child2.sequence)}")
        return child1, child2

    def mutation(self, individual):
        """Performs PLM-guided mutation."""
        mutated_sequence = self.plm.perform_plm_mutation(individual.sequence)
        return Individual(mutated_sequence)

    def run(self):
        """Main evolutionary loop."""
        # 1. Initialization
        initial_sequences = self.plm.initialize_population()
        self.population = [Individual(seq) for seq in initial_sequences]
        
        for gen in range(self.config['ea']['num_generations']):
            logging.info(f"\n{'='*20} GENERATION {gen+1}/{self.config['ea']['num_generations']} {'='*20}")
            
            # 2. Evaluation
            self.evaluate_population()
            
            # Track best individual
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best
                logging.info(f"New best individual found with fitness {self.best_individual.fitness:.4f}")
                if (gen + 1) % self.config['output']['save_interval'] == 0:
                    save_path = os.path.join(self.config['output']['results_dir'], f"gen_{gen+1}_best.pdb")
                    save_pdb_string(self.best_individual.pdb_string, save_path)
                    
                    # Save sequence information
                    seq_save_path = os.path.join(self.config['output']['results_dir'], f"gen_{gen+1}_sequences.txt")
                    self.save_generation_sequences(seq_save_path, gen + 1)

            logging.info(f"Generation {gen+1} Summary: Best Fitness = {self.best_individual.fitness:.4f}, Pop Avg Fitness = {np.mean([ind.fitness for ind in self.population]):.4f}")

            # 3. Selection
            parents = self.selection()
            
            # 4. Reproduction
            next_generation = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if (i+1) < len(parents) else parents
                
                # Crossover
                if random.random() < self.config['ea']['crossover_rate']:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = Individual(parent1.sequence), Individual(parent2.sequence)
                
                # Mutation
                if random.random() < self.config['ea']['mutation_rate']:
                    child1 = self.mutation(child1)
                if random.random() < self.config['ea']['mutation_rate']:
                    child2 = self.mutation(child2)
                
                next_generation.extend([child1, child2])
            
            self.population = next_generation[:self.config['ea']['population_size']]

        logging.info("Evolutionary run completed.")
        logging.info(f"Final best individual: {self.best_individual}")
        final_save_path = os.path.join(self.config['output']['results_dir'], "final_best.pdb")
        save_pdb_string(self.best_individual.pdb_string, final_save_path)
        
        return self.best_individual

    def save_generation_sequences(self, filepath, generation):
        """Save all sequences from current generation to a text file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(f"Generation {generation} Sequences\n")
            f.write("=" * 50 + "\n\n")
            for i, ind in enumerate(self.population):
                f.write(f"Individual {i+1}:\n")
                f.write(f"  Sequence: {ind.sequence}\n")
                f.write(f"  Length: {len(ind.sequence)}\n")
                f.write(f"  F_seq: {ind.f_seq:.3f}\n")
                f.write(f"  F_struct: {ind.f_struct:.3f}\n")
                f.write(f"  F_gnn: {ind.f_gnn:.3f}\n")
                f.write(f"  Total fitness: {ind.fitness:.4f}\n")
                f.write("-" * 30 + "\n")
        logging.info(f"Saved generation {generation} sequences to {filepath}")

if __name__ == '__main__':
    
    config = load_config()
    evo_system = EvoGraphPLM(config)
    best_protein = evo_system.run()