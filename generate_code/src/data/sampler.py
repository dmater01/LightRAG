import random
import logging
from typing import List, Dict, Any, Optional
from src.data.loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TaskSampler:
    """
    Responsible for sampling pairs of (Target, Injected) inputs for the benchmark.
    Ensures that for classification tasks, the ground truth labels of the target
    and injected inputs are different to allow for clear attribution of the model's output.
    """
    
    def __init__(self, config_path: str = "config/tasks.yaml"):
        self.loader = DataLoader(config_path)
        self.cache = {}  # Cache loaded datasets to avoid reloading
        
    def _get_data(self, task_name: str) -> List[Dict[str, Any]]:
        """Helper to load data with caching."""
        if task_name not in self.cache:
            logger.info(f"Loading data for task: {task_name}")
            # We load a larger pool to ensure we can find enough non-overlapping pairs
            # Loading 1000 should be sufficient for 100 pairs
            self.cache[task_name] = self.loader.load_task_data(task_name, split="validation", limit=1000)
        return self.cache[task_name]

    def sample_pairs(self, 
                     target_task: str, 
                     injected_task: str, 
                     num_samples: int = 100,
                     seed: int = 42) -> List[Dict[str, Any]]:
        """
        Sample pairs of (target_sample, injected_sample).
        
        Args:
            target_task: Name of the target task (e.g., 'sst2')
            injected_task: Name of the injected task (e.g., 'sms')
            num_samples: Number of pairs to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of dictionaries containing:
                {
                    'target': { ... target sample dict ... },
                    'injected': { ... injected sample dict ... }
                }
        """
        random.seed(seed)
        
        target_data = self._get_data(target_task)
        injected_data = self._get_data(injected_task)
        
        if not target_data or not injected_data:
            logger.error(f"Failed to load data for {target_task} or {injected_task}")
            return []
            
        pairs = []
        attempts = 0
        max_attempts = num_samples * 50  # Prevent infinite loops
        
        # Create indices to sample from
        t_indices = list(range(len(target_data)))
        i_indices = list(range(len(injected_data)))
        
        random.shuffle(t_indices)
        random.shuffle(i_indices)
        
        t_ptr = 0
        i_ptr = 0
        
        while len(pairs) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Reset pointers and reshuffle if we run out of data
            if t_ptr >= len(t_indices):
                random.shuffle(t_indices)
                t_ptr = 0
            if i_ptr >= len(i_indices):
                random.shuffle(i_indices)
                i_ptr = 0
                
            t_idx = t_indices[t_ptr]
            i_idx = i_indices[i_ptr]
            
            target_sample = target_data[t_idx]
            injected_sample = injected_data[i_idx]
            
            # Check label constraint
            # We only strictly enforce this if the labels are strings and identical
            # This is critical for the diagonal of the matrix (same task)
            # or if tasks happen to share label space (e.g. binary yes/no)
            
            t_label = target_sample.get('label')
            i_label = injected_sample.get('label')
            
            # If labels are identical, we skip this pair to avoid ambiguity
            # This handles the "Target Label != Injected Label" requirement
            if t_label is not None and i_label is not None and t_label == i_label:
                # Advance one of the pointers to try a different combination
                # We advance the injected pointer to try a different injected sample for this target
                i_ptr += 1
                continue
                
            pairs.append({
                'target': target_sample,
                'injected': injected_sample
            })
            
            # Advance both pointers
            t_ptr += 1
            i_ptr += 1
            
        if len(pairs) < num_samples:
            logger.warning(f"Could only sample {len(pairs)} pairs for {target_task} -> {injected_task} after {attempts} attempts.")
            
        logger.info(f"Successfully sampled {len(pairs)} pairs for {target_task} -> {injected_task}")
        return pairs
