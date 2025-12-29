#!/usr/bin/env python3
"""
Defense Calibration Script

This script calculates the Perplexity (PPL) thresholds for detection-based defenses.
It runs the specified LLM on clean data from the validation split of the tasks
and determines the threshold T such that the False Positive Rate (FPR) is <= target (e.g., 1%).

Usage:
    python scripts/calibrate_defense.py --tasks all --model_type local --fpr 0.01
"""

import argparse
import os
import sys
import yaml
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.core.llm_wrapper import LLMWrapper
from src.data.loader import DataLoader
from src.evaluation.calibrator import ThresholdCalibrator

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate PPL detection thresholds")
    
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="all", 
        help="Comma-separated list of tasks to calibrate (or 'all')"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="local", 
        choices=["local", "openai"],
        help="Type of model to use for PPL calculation (usually local)"
    )
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="config/models.yaml", 
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--task_config", 
        type=str, 
        default="config/tasks.yaml", 
        help="Path to task configuration file"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="config/defense_thresholds.json", 
        help="Path to save calibrated thresholds"
    )
    parser.add_argument(
        "--fpr", 
        type=float, 
        default=0.01, 
        help="Target False Positive Rate (default: 0.01 for 1%)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100, 
        help="Number of samples to use for calibration per task"
    )
    
    return parser.parse_args()

def get_task_list(task_arg: str, config_path: str) -> List[str]:
    """Resolve task argument to list of task names."""
    if task_arg.lower() == "all":
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return list(config.keys())
    else:
        return [t.strip() for t in task_arg.split(",")]

def main():
    args = parse_args()
    logger = setup_logger("calibration")
    
    logger.info(f"Starting calibration with FPR target: {args.fpr}")
    logger.info(f"Model Type: {args.model_type}")
    
    # 1. Initialize Components
    try:
        # Initialize LLM Wrapper
        # Note: PPL detection usually requires a local model to access logits/perplexity
        llm = LLMWrapper(model_type=args.model_type, config_path=args.model_config)
        
        # Initialize Data Loader
        data_loader = DataLoader(config_path=args.task_config)
        
        # Initialize Calibrator
        calibrator = ThresholdCalibrator(llm_wrapper=llm, data_loader=data_loader)
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
        
    # 2. Determine Tasks
    try:
        tasks = get_task_list(args.tasks, args.task_config)
        logger.info(f"Tasks to calibrate: {tasks}")
    except Exception as e:
        logger.error(f"Failed to load task configuration: {e}")
        sys.exit(1)
        
    # 3. Run Calibration
    try:
        thresholds = calibrator.calibrate(
            tasks=tasks,
            limit=args.limit,
            fpr_target=args.fpr
        )
        
        # 4. Save Results
        calibrator.save_thresholds(thresholds, args.output_path)
        logger.info(f"Calibration complete. Thresholds saved to {args.output_path}")
        
        # Log a summary
        for task, values in thresholds.items():
            logger.info(f"Task: {task} | PPL Threshold: {values.get('ppl_threshold', 'N/A'):.2f} | Window PPL Threshold: {values.get('window_ppl_threshold', 'N/A'):.2f}")
            
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
