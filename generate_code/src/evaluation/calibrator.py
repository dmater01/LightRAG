import numpy as np
import json
import os
from typing import List, Dict, Any
from src.utils.logger import setup_logger

class ThresholdCalibrator:
    """
    Calibrates Perplexity (PPL) thresholds for detection defenses.
    Determines the threshold T such that the False Positive Rate (FPR) 
    on clean data is <= target_fpr (default 1%).
    """
    
    def __init__(self, llm_wrapper, data_loader):
        """
        Args:
            llm_wrapper: Instance of LLMWrapper to calculate perplexity.
            data_loader: Instance of DataLoader to fetch clean data.
        """
        self.llm = llm_wrapper
        self.data_loader = data_loader
        self.logger = setup_logger("calibrator")

    def calibrate(self, tasks: List[str], limit: int = 100, fpr_target: float = 0.01) -> Dict[str, Dict[str, float]]:
        """
        Calibrates thresholds for the specified tasks.
        
        Args:
            tasks: List of task names to calibrate.
            limit: Number of samples to use per task.
            fpr_target: Target False Positive Rate (default 0.01 for 1%).
            
        Returns:
            Dictionary mapping task names to threshold dictionaries:
            {
                'task_name': {
                    'ppl_threshold': float,
                    'window_ppl_threshold': float
                }
            }
        """
        thresholds = {}
        
        self.logger.info(f"Starting calibration for {len(tasks)} tasks with target FPR {fpr_target}")
        
        for task in tasks:
            self.logger.info(f"Calibrating for task: {task}")
            try:
                # Load clean data
                # We use the validation split as 'clean' data for calibration
                data = self.data_loader.load_task_data(task, split="validation", limit=limit)
                
                if not data:
                    self.logger.warning(f"No data found for task {task}. Skipping.")
                    continue
                
                ppl_scores = []
                window_ppl_scores = []
                
                # Calculate PPL for each sample
                for i, item in enumerate(data):
                    text = item['formatted_input']
                    
                    try:
                        # Calculate standard PPL
                        ppl = self.llm.get_perplexity(text)
                        if ppl is not None:
                            ppl_scores.append(ppl)
                            
                        # Calculate Windowed PPL
                        w_ppl = self.llm.get_windowed_perplexity(text)
                        if w_ppl is not None:
                            window_ppl_scores.append(w_ppl)
                            
                    except Exception as e:
                        self.logger.debug(f"Error calculating PPL for sample {i} in {task}: {e}")
                
                if not ppl_scores:
                    self.logger.warning(f"No valid PPL scores computed for {task}")
                    thresholds[task] = {'ppl_threshold': 10000.0, 'window_ppl_threshold': 10000.0}
                    continue
                
                # Calculate thresholds
                # We want the value at the (1 - fpr_target) percentile
                # e.g., for 1% FPR, we want the 99th percentile value.
                # Any value above this will be flagged as malicious.
                percentile = (1.0 - fpr_target) * 100
                
                t_ppl = float(np.percentile(ppl_scores, percentile))
                t_w_ppl = float(np.percentile(window_ppl_scores, percentile))
                
                thresholds[task] = {
                    'ppl_threshold': t_ppl,
                    'window_ppl_threshold': t_w_ppl
                }
                
                self.logger.info(f"Task {task} calibrated - PPL T: {t_ppl:.2f}, Window PPL T: {t_w_ppl:.2f}")
                
            except Exception as e:
                self.logger.error(f"Failed to calibrate task {task}: {e}")
                thresholds[task] = {'ppl_threshold': 10000.0, 'window_ppl_threshold': 10000.0}
        
        return thresholds

    def save_thresholds(self, thresholds: Dict[str, Dict[str, float]], filepath: str = "config/defense_thresholds.json"):
        """Saves calibrated thresholds to a JSON file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(thresholds, f, indent=2)
            self.logger.info(f"Thresholds saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save thresholds to {filepath}: {e}")

    def load_thresholds(self, filepath: str = "config/defense_thresholds.json") -> Dict[str, Dict[str, float]]:
        """Loads thresholds from a JSON file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Threshold file {filepath} not found.")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load thresholds from {filepath}: {e}")
            return {}
