import yaml
import os
from datasets import load_dataset
from src.utils.logger import setup_logger
from src.utils.text_processing import clean_text

logger = setup_logger(__name__)

class DataLoader:
    """
    Handles loading and formatting of datasets for the 7 benchmark tasks.
    """
    def __init__(self, config_path="config/tasks.yaml"):
        self.config_path = config_path
        self.tasks_config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_task_data(self, task_name, split="validation", limit=None):
        """
        Loads data for a specific task.
        
        Args:
            task_name (str): Name of the task (e.g., 'mrpc', 'sst2').
            split (str): Dataset split to load (default: 'validation').
            limit (int): Maximum number of examples to load.
            
        Returns:
            list: List of dictionaries containing 'original_inputs', 'formatted_input', and 'label'.
        """
        if task_name not in self.tasks_config:
            raise ValueError(f"Task {task_name} not found in config.")
        
        task_conf = self.tasks_config[task_name]
        dataset_name = task_conf['dataset_name']
        subset = task_conf.get('subset')
        
        logger.info(f"Loading dataset {dataset_name} (subset={subset}) for task {task_name}...")
        
        try:
            # Determine correct split
            load_split = split
            
            # Handle dataset-specific split availability
            if task_name == 'hsol' and split == 'validation':
                # hate_speech_offensive only has 'train', so we use a slice of it or just load train
                # For reproduction, we'll use 'train' if validation isn't requested explicitly or fallback
                load_split = 'train'
            elif task_name == 'sms' and split == 'validation':
                # sms_spam usually has 'train'
                load_split = 'train'
            elif task_name == 'jfleg' and split == 'validation':
                load_split = 'validation'
            elif task_name == 'gigaword' and split == 'validation':
                load_split = 'validation'
                
            # Load dataset
            if subset:
                ds = load_dataset(dataset_name, subset, split=load_split)
            else:
                ds = load_dataset(dataset_name, split=load_split)
            
            # Apply limit
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
                
            data = []
            for item in ds:
                processed_item = self._process_item(item, task_conf, task_name)
                if processed_item:
                    data.append(processed_item)
            
            logger.info(f"Successfully loaded {len(data)} examples for {task_name}.")
            return data

        except Exception as e:
            logger.error(f"Error loading dataset for {task_name}: {e}")
            raise e

    def _process_item(self, item, task_conf, task_name):
        # Extract input fields
        input_fields = task_conf['input_fields']
        input_values = {}
        
        for field in input_fields:
            if field not in item:
                logger.warning(f"Field {field} missing in item for task {task_name}")
                return None
            input_values[field] = clean_text(str(item[field]))
            
        # Format prompt
        prompt_template = task_conf['prompt_template']
        try:
            formatted_input = prompt_template.format(**input_values)
        except KeyError as e:
            logger.error(f"KeyError formatting prompt: {e}. Available keys: {input_values.keys()}")
            return None

        # Process Label
        label_text = self._extract_label(item, task_conf, task_name)
        
        return {
            'original_inputs': input_values,
            'formatted_input': formatted_input,
            'label': label_text
        }

    def _extract_label(self, item, task_conf, task_name):
        """
        Extracts and formats the label/target from the dataset item.
        """
        label_map = task_conf.get('label_map')
        
        # Case 1: Classification tasks with label map
        if label_map:
            if 'label' in item:
                label_val = item['label']
                if label_val in label_map:
                    return label_map[label_val]
                else:
                    # Handle string labels if dataset returns strings instead of ints
                    # or if label_map keys are strings
                    return str(label_val)
            else:
                logger.warning(f"Label column missing for classification task {task_name}")
                return "Unknown"

        # Case 2: Generation tasks or tasks without label map
        # Identify target column based on task/dataset
        if task_name == 'jfleg':
            # Jfleg has 'corrections' which is a list of strings
            if 'corrections' in item and len(item['corrections']) > 0:
                return item['corrections'][0]
        elif task_name == 'gigaword':
            if 'summary' in item:
                return item['summary']
        
        # Fallback to 'label' if it exists
        if 'label' in item:
            return str(item['label'])
            
        # Fallback to 'text_label' or similar if known
        return ""
