# plm_tools.py
# Functions for interacting with Protein Language Models (PLMs)

import torch
import numpy as np
import logging
import random  # Add missing import
import torch.nn as nn # Add this import for the loss function
from transformers import AutoModelForCausalLM, AutoTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.models.esm3 import ESM3

class PLMTools:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        logging.info("Initializing PLM Tools...")
        
        # Generator model (ProtGPT2)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(config['models']['plm']['generator_model'])
        # Fix: Set pad_token to eos_token for GPT-style models
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            logging.info("Set pad_token to eos_token for ProtGPT2 tokenizer")
        
        self.generator_model = AutoModelForCausalLM.from_pretrained(config['models']['plm']['generator_model']).to(device)
        
        # Evaluator and Mutator model (ESM-3)
        self.evaluator_tokenizer = EsmSequenceTokenizer()
        self.evaluator_model = ESM3.from_pretrained(config['models']['plm']['evaluator_model']).float().to(device)

        logging.info("PLM Tools initialized successfully.")

    def initialize_population(self):
        """Generates an initial population of protein sequences using ProtGPT2."""
        pop_size = self.config['ea']['population_size']
        min_len = self.config['initial_population']['min_length']
        max_len = self.config['initial_population']['max_length']
        prompt = self.config['initial_population']['prompt']
        
        model_max_len = self.generator_model.config.n_positions
        if max_len > model_max_len:
            logging.warning(f"Config max_length ({max_len}) exceeds model limit ({model_max_len}). Capping at {model_max_len}.")
            max_len = model_max_len
        

        #input_ids = self.generator_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        population = []
        logging.info(f"Generating initial population of size {pop_size}...")
        
        # Encode prompt once to get its length
        prompt_inputs = self.generator_tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        prompt_length = prompt_inputs['input_ids'].shape[1]
        
        max_attempts = pop_size * 5  # Prevent infinite loops
        attempts = 0
        
        while len(population) < pop_size and attempts < max_attempts:
            attempts += 1
            
            # Calculate target total length (prompt + generated sequence)
            target_gen_len = np.random.randint(min_len, max_len + 1)  # Random target within range
            
            # Generate with dynamic length control
            with torch.no_grad():
                outputs = self.generator_model.generate(
                    input_ids=prompt_inputs['input_ids'],
                    attention_mask=prompt_inputs['attention_mask'],
                    max_new_tokens=target_gen_len + 20,  # Use max_new_tokens instead of max_length
                    min_new_tokens=max(1, min_len - 20),  # Minimum new tokens to generate, ensure >= 1
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.generator_tokenizer.pad_token_id,
                    eos_token_id=self.generator_tokenizer.eos_token_id,
                    temperature=1.0,  # Higher temperature for more diversity
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Extract only the generated part (remove prompt)
            generated_tokens = outputs[0][prompt_length:].tolist()
            sequence = self.generator_tokenizer.decode(generated_tokens, skip_special_tokens=True).replace(" ", "")
            
            # Clean up the sequence - keep only valid amino acids
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            clean_sequence = "".join([char for char in sequence.upper() if char in valid_aas])
            
            # Accept sequences within range OR truncate if too long
            if len(clean_sequence) >= min_len:
                if len(clean_sequence) > max_len:
                    # Truncate to max length
                    clean_sequence = clean_sequence[:max_len]
                    logging.info(f"Truncated sequence from {len(sequence)} to {max_len}")
                
                population.append(clean_sequence)
                logging.info(f"Generated sequence {len(population)}/{pop_size}: length {len(clean_sequence)}")
            
            elif attempts % 20 == 0:  # Log progress every 20 attempts
                logging.info(f"Attempt {attempts}: Generated sequence length {len(clean_sequence)}, target range [{min_len}, {max_len}]")
        
        # If we couldn't generate enough sequences, fill with truncated/padded versions
        if len(population) < pop_size:
            logging.warning(f"Only generated {len(population)}/{pop_size} sequences. Filling remaining with variations...")
            
            while len(population) < pop_size:
                # Take a random existing sequence and modify it
                if population:
                    base_seq = random.choice(population)
                    # Add some random mutations to create diversity
                    modified_seq = self._modify_sequence_length(base_seq, min_len, max_len)
                    population.append(modified_seq)
                else:
                    # Fallback: create a random sequence
                    random_seq = self._create_random_sequence(np.random.randint(min_len, max_len + 1))
                    population.append(random_seq)
        
        return population

    def _modify_sequence_length(self, sequence, min_len, max_len):
        """Modify a sequence to fit within the target length range."""
        current_len = len(sequence)
        target_len = np.random.randint(min_len, max_len + 1)
        
        if current_len > target_len:
            # Truncate randomly from start or end
            if np.random.random() > 0.5:
                return sequence[:target_len]
            else:
                return sequence[-target_len:]
        elif current_len < target_len:
            # Extend with random amino acids
            additional_aas = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 
                                            size=target_len - current_len)
            return sequence + "".join(additional_aas)
        else:
            return sequence

    def _create_random_sequence(self, length):
        """Create a random protein sequence of specified length."""
        return "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=length))

    def calculate_f_seq(self, sequences):
        """Calculates the sequence fitness (F_seq) as the pseudo-log-likelihood."""
        f_seq_scores = []
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for seq in sequences:
                # Tokenize the sequence using the ESM3 native tokenizer
                token_ids = self.evaluator_tokenizer([seq], return_tensors="pt")['input_ids'].to(self.device)
                
                # Get logits from the model using the correct argument name
                outputs = self.evaluator_model(sequence_tokens=token_ids)
                logits = outputs.sequence_logits
                
                # Manually calculate the cross-entropy loss
                # Reshape logits and labels for the loss function
                vocab_size = logits.size(-1)
                logits_flat = logits.view(-1, vocab_size)
                # The input tokens are the labels for calculating perplexity
                labels_flat = token_ids.view(-1)
                
                loss = loss_fn(logits_flat, labels_flat)
                
                # The loss is the negative log-likelihood. We want to maximize fitness,
                # so we take the negative of the loss.
                neg_log_likelihood = loss.item()
                f_seq_scores.append(-neg_log_likelihood)
        
        return np.array(f_seq_scores)

    def perform_plm_mutation(self, sequence):
        """Performs a single 'smart' mutation on a sequence using ESM-3's MLM head."""
        seq_list = list(sequence)
        if not seq_list:
            return sequence

        position = np.random.randint(len(seq_list))
        original_residue = seq_list[position]
        seq_list[position] = '<mask>'
        
        # Tokenize the masked sequence
        token_ids = self.evaluator_tokenizer(["".join(seq_list)], return_tensors="pt")['input_ids'].to(self.device)

        with torch.no_grad():
            outputs = self.evaluator_model(sequence_tokens=token_ids)
            logits = outputs.sequence_logits

        mask_token_index = torch.where(token_ids == self.evaluator_tokenizer.mask_token_id)[1]
        mask_logits = logits[0, mask_token_index, :]
        
        probabilities = torch.softmax(mask_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probabilities, k=10, dim=-1)
        
        new_residue = original_residue
        for idx in top_k_indices.squeeze():
            token = self.evaluator_tokenizer.decode(idx.item())
            # Ensure the token is a valid, single-character amino acid
            if token and len(token) == 1 and token.isupper() and token in "ACDEFGHIKLMNPQRSTVWY":
                if token != original_residue:
                    new_residue = token
                    break
        
        if new_residue == original_residue:
            return sequence

        mutated_seq_list = list(sequence)
        mutated_seq_list[position] = new_residue
        mutated_sequence = "".join(mutated_seq_list)
        
        logging.info(f"PLM Mutation: {sequence} -> {mutated_sequence} at position {position}")
        return mutated_sequence