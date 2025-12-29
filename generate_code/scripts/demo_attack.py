import argparse
import sys
import os
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.attacks import get_attack_strategy, ATTACK_REGISTRY
from src.core.defenses import get_defense_strategy, DEFENSE_REGISTRY
from src.core.llm_wrapper import LLMWrapper
from src.data.sampler import TaskSampler
from src.utils.logger import setup_logger

def load_task_config(config_path="config/tasks.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Demonstrate a single prompt injection attack instance.")
    
    parser.add_argument("--target_task", type=str, default="sst2", help="Target task (benign)")
    parser.add_argument("--injected_task", type=str, default="sms", help="Injected task (malicious)")
    parser.add_argument("--attack", type=str, default="combined", choices=list(ATTACK_REGISTRY.keys()), help="Attack strategy")
    parser.add_argument("--defense", type=str, default="none", choices=["none"] + list(DEFENSE_REGISTRY.keys()), help="Defense strategy")
    parser.add_argument("--model_type", type=str, default="local", choices=["local", "openai"], help="Model backend")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    logger = setup_logger("demo_attack")
    logger.info(f"Starting demo with Target={args.target_task}, Injected={args.injected_task}, Attack={args.attack}, Defense={args.defense}")

    # 1. Load Data
    logger.info("Loading data sampler...")
    try:
        sampler = TaskSampler()
        # Sample just 1 pair
        pairs = sampler.sample_pairs(args.target_task, args.injected_task, num_samples=1, seed=args.seed)
        if not pairs:
            logger.error("Failed to sample a pair. Check task names and data availability.")
            return
        
        sample = pairs[0]
        target_sample = sample['target']
        injected_sample = sample['injected']
        
        # Load task config to get prompts
        tasks_config = load_task_config()
        target_prompt_template = tasks_config[args.target_task]['prompt_template']
        
        # Extract components
        # Target instruction (s_t) is part of the prompt template usually, but here we treat the template as s_t
        # Target input (x_t)
        x_t = target_sample['formatted_input']
        
        # Injected instruction (s_e)
        # For the injected task, we need its instruction. 
        # In the paper framework: x_tilde = A(x_t, s_e, x_e)
        # s_e is the instruction of the injected task.
        injected_prompt_template = tasks_config[args.injected_task]['prompt_template']
        # We usually assume the instruction is the template without the input placeholder, 
        # or we just use the template with the input filled as the "payload".
        # However, the attack signature is A(x_t, s_e, x_e).
        # Let's approximate s_e as the prompt template text before the placeholder.
        # A better way given the loader structure:
        # The loader returns 'formatted_input' which is just the input text from the dataset.
        # The prompt template is separate.
        
        s_e = injected_prompt_template.replace("{text}", "").strip() # Simplified extraction
        x_e = injected_sample['formatted_input']
        
        logger.info("Data loaded successfully.")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # 2. Initialize Model
    logger.info(f"Initializing {args.model_type} model...")
    try:
        llm = LLMWrapper(model_type=args.model_type)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # 3. Apply Attack
    logger.info(f"Applying {args.attack} attack...")
    attack_strategy = get_attack_strategy(args.attack)
    
    # x_tilde = A(x_t, s_e, x_e)
    x_tilde = attack_strategy.apply(target_input=x_t, injected_instruction=s_e, injected_input=x_e)
    
    # 4. Apply Defense (if any)
    final_prompt = ""
    defense_detected = False
    
    # The final prompt sent to LLM usually follows the structure: s_t + x_tilde
    # But defenses might modify this.
    
    # Construct initial full prompt (Target Instruction + Attacked Input)
    # Note: simple concatenation s_t + " " + x_tilde
    target_instruction = target_prompt_template.replace("{text}", "").strip()
    
    if args.defense != "none":
        logger.info(f"Applying {args.defense} defense...")
        defense_strategy = get_defense_strategy(args.defense)
        
        # Defenses might need the LLM (e.g. Paraphrasing, PPL)
        defense_result = defense_strategy.apply(
            target_instruction=target_instruction,
            input_text=x_tilde,
            llm=llm
        )
        
        defense_detected = defense_result.get('is_detected', False)
        final_prompt = defense_result.get('prompt', "")
        
        if defense_detected:
            logger.warning(f"Defense {args.defense} DETECTED the attack!")
        else:
            logger.info(f"Defense {args.defense} did NOT detect the attack.")
            
    else:
        # No defense: Standard prompt construction
        # s_t + x_tilde
        # We use the template logic: replace {text} with x_tilde
        final_prompt = target_prompt_template.replace("{text}", x_tilde)

    # 5. Query Model
    logger.info("Querying model...")
    if defense_detected and args.defense in ["ppl_detection", "windowed_ppl_detection", "known_answer"]:
        # For detection defenses, if detected, we usually block the request or return a canned response.
        # The paper implies detection flags it. We can simulate a "Blocked" response.
        response = "[BLOCKED BY DEFENSE]"
    else:
        response = llm.query(final_prompt)

    # 6. Display Results
    print("\n" + "="*60)
    print(f"DEMO RESULTS: {args.attack} Attack on {args.target_task} (Target) with {args.injected_task} (Injected)")
    print("="*60)
    
    print(f"\n[1] Target Input (x_t) [{args.target_task}]:")
    print("-" * 20)
    print(x_t)
    
    print(f"\n[2] Injected Instruction (s_e) [{args.injected_task}]:")
    print("-" * 20)
    print(s_e)
    
    print(f"\n[3] Injected Input (x_e) [{args.injected_task}]:")
    print("-" * 20)
    print(x_e)
    
    print(f"\n[4] Attacked Input (x_tilde):")
    print("-" * 20)
    print(x_tilde)
    
    print(f"\n[5] Final Prompt to LLM (after defense):")
    print("-" * 20)
    print(final_prompt)
    
    print(f"\n[6] Model Output:")
    print("-" * 20)
    print(response)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
