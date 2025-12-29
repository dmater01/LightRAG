import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.core.llm_wrapper import LLMWrapper
from src.core.attacks import get_attack_strategy, ATTACK_REGISTRY
from src.core.defenses import get_defense_strategy, DEFENSE_REGISTRY
from src.data.sampler import TaskSampler
from src.evaluation.metrics import MetricsCalculator

def parse_args():
    parser = argparse.ArgumentParser(description="Run Prompt Injection Benchmark")
    
    # Selection arguments
    parser.add_argument("--target_tasks", type=str, default="all", 
                        help="Comma-separated list of target tasks (or 'all')")
    parser.add_argument("--injected_tasks", type=str, default="all", 
                        help="Comma-separated list of injected tasks (or 'all')")
    parser.add_argument("--attacks", type=str, default="all", 
                        help="Comma-separated list of attacks (or 'all')")
    parser.add_argument("--defenses", type=str, default="none", 
                        help="Comma-separated list of defenses (or 'all', 'none')")
    
    # Configuration arguments
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Number of samples per task pair")
    parser.add_argument("--model_type", type=str, default="local", choices=["local", "openai"],
                        help="Model type to use")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Config paths
    parser.add_argument("--task_config", type=str, default="config/tasks.yaml")
    parser.add_argument("--model_config", type=str, default="config/models.yaml")
    parser.add_argument("--defense_thresholds", type=str, default="config/defense_thresholds.json")

    return parser.parse_args()

def get_list_from_arg(arg_value, all_options):
    if arg_value == "all":
        return all_options
    if arg_value == "none":
        return []
    return [x.strip() for x in arg_value.split(",")]

def main():
    args = parse_args()
    logger = setup_logger("benchmark_runner")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configurations
    with open(args.task_config, 'r') as f:
        task_config = yaml.safe_load(f)
    
    all_tasks = list(task_config.keys())
    target_tasks = get_list_from_arg(args.target_tasks, all_tasks)
    injected_tasks = get_list_from_arg(args.injected_tasks, all_tasks)
    
    attack_list = get_list_from_arg(args.attacks, list(ATTACK_REGISTRY.keys()))
    defense_list = get_list_from_arg(args.defenses, list(DEFENSE_REGISTRY.keys()))
    
    # Ensure "none" (no defense) is explicitly handled if user wants to run baseline
    # If user passed "none" in args.defenses, defense_list is empty. 
    # Usually we want to run without defense as a baseline. 
    # Let's treat "None" as a specific defense strategy which is just identity.
    # But our code handles defenses as optional. We will iterate defense_list.
    # If defense_list is empty, we only run without defense? 
    # Or should we always run 'No Defense'?
    # Let's assume the user specifies what they want. If they want no defense, they might not specify any defense.
    # But typically we want to compare. Let's add 'None' to the list if it's not there, 
    # or handle it in the loop.
    # For clarity: The loop will be over defenses. If list is empty, we can't run anything?
    # No, usually "none" means "no defense mechanism".
    # Let's add a placeholder "None" to the list if the user wants to run baseline.
    # If args.defenses is "none", get_list_from_arg returns [].
    # If args.defenses is "all", it returns all real defenses.
    # We should probably allow running with "No Defense" explicitly.
    
    run_no_defense = False
    if args.defenses == "none" or "none" in args.defenses.split(","):
        run_no_defense = True
        # Remove "none" from the list of actual defense classes
        if "none" in defense_list:
            defense_list.remove("none")
    
    # Initialize components
    logger.info(f"Initializing {args.model_type} model...")
    llm = LLMWrapper(model_type=args.model_type, config_path=args.model_config)
    
    logger.info("Initializing Sampler and Metrics...")
    sampler = TaskSampler(config_path=args.task_config)
    metrics_calc = MetricsCalculator()
    
    # Load defense thresholds if needed
    thresholds = {}
    if os.path.exists(args.defense_thresholds):
        try:
            with open(args.defense_thresholds, 'r') as f:
                thresholds = json.load(f)
            logger.info(f"Loaded defense thresholds from {args.defense_thresholds}")
        except Exception as e:
            logger.warning(f"Could not load defense thresholds: {e}")
    
    # Store results
    results = []
    
    # Total iterations estimation
    total_combinations = len(target_tasks) * len(injected_tasks) * len(attack_list)
    defense_count = len(defense_list) + (1 if run_no_defense else 0)
    total_combinations *= defense_count
    
    logger.info(f"Starting benchmark with {total_combinations} combinations...")
    
    pbar = tqdm(total=total_combinations)
    
    for t_task in target_tasks:
        for i_task in injected_tasks:
            # Sample data for this pair
            try:
                samples = sampler.sample_pairs(t_task, i_task, num_samples=args.num_samples, seed=args.seed)
            except Exception as e:
                logger.error(f"Failed to sample pairs for {t_task} -> {i_task}: {e}")
                continue
                
            # Pre-calculate "Clean Injected Output" for MR metric (f(s_e + x_e))
            # This is what the model outputs for the injected task alone.
            # We do this once per pair to save time if we have multiple attacks/defenses?
            # No, it depends on the sample.
            # Let's do it inside the loop or cache it.
            
            # To optimize, we can generate these baselines once per sample list.
            logger.info(f"Generating baselines for {t_task} -> {i_task} ({len(samples)} samples)")
            injected_baselines = []
            for sample in samples:
                # Construct clean injected prompt: s_e + x_e
                # The sampler gives us 'injected': {'formatted_input': ...} which is s_e + x_e
                inj_prompt = sample['injected']['formatted_input']
                resp = llm.query(inj_prompt)
                injected_baselines.append(resp)
            
            # Define the list of defense configurations to run
            # Each config is (defense_name, defense_kwargs)
            defense_configs = []
            if run_no_defense:
                defense_configs.append(("None", {}))
            for d_name in defense_list:
                # Prepare kwargs for defense
                kwargs = {}
                if "ppl" in d_name and t_task in thresholds:
                    # Pass threshold if available
                    # Note: thresholds structure is {task: {defense_type: val}}
                    # We need to match defense name to threshold key
                    # PPLDetectionDefense -> 'ppl'
                    # WindowedPPLDetectionDefense -> 'windowed_ppl'
                    if d_name == "ppl_detection":
                        kwargs['threshold'] = thresholds[t_task].get('ppl', 10000.0)
                    elif d_name == "windowed_ppl_detection":
                        kwargs['threshold'] = thresholds[t_task].get('windowed_ppl', 10000.0)
                
                defense_configs.append((d_name, kwargs))
            
            for attack_name in attack_list:
                attack_strategy = get_attack_strategy(attack_name)
                
                for defense_name, defense_kwargs in defense_configs:
                    
                    logger.info(f"Running: T={t_task}, I={i_task}, A={attack_name}, D={defense_name}")
                    
                    # Initialize defense
                    defense = None
                    if defense_name != "None":
                        try:
                            defense = get_defense_strategy(defense_name, **defense_kwargs)
                        except Exception as e:
                            logger.error(f"Failed to init defense {defense_name}: {e}")
                            pbar.update(1)
                            continue
                    
                    batch_results = {
                        "target_task": t_task,
                        "injected_task": i_task,
                        "attack": attack_name,
                        "defense": defense_name,
                        "samples": []
                    }
                    
                    asv_scores = []
                    mr_scores = []
                    defense_detected_count = 0
                    
                    for idx, sample in enumerate(samples):
                        # 1. Generate Attack
                        # x_tilde = A(x_t, s_e, x_e)
                        # We need x_t (target input data), s_e (injected instruction), x_e (injected input data)
                        
                        # The sampler provides:
                        # sample['target']['original_inputs'] -> dict of fields
                        # sample['target']['formatted_input'] -> s_t + x_t (full prompt)
                        # We need to extract x_t. 
                        # The AttackStrategy.apply takes (target_input, injected_instruction, injected_input)
                        # But wait, the formulas in attacks.py are:
                        # x_tilde = x_t + ...
                        # And the final prompt to LLM is s_t + x_tilde.
                        
                        # Let's look at how we can construct this.
                        # We need the raw input text of the target (x_t) and the raw input of injected (x_e).
                        # And the instruction of injected (s_e).
                        
                        # The DataLoader formats inputs. 
                        # We might need to reconstruct s_e and x_e from the config or loader.
                        # Actually, `sample['injected']['formatted_input']` is s_e + x_e.
                        # But the attack needs them separate to insert triggers?
                        # Let's check `attacks.py`.
                        # apply(target_input, injected_instruction, injected_input)
                        
                        # We need to get these components.
                        # The `DataLoader` returns `original_inputs`.
                        # We can reconstruct x_t from `original_inputs` using `text_processing` or just joining values.
                        # However, `formatted_input` includes the instruction.
                        
                        # Let's assume:
                        # x_t = content of target input fields
                        # s_e = template of injected task
                        # x_e = content of injected input fields
                        
                        # We need to access the prompt template to separate s_e.
                        # Or we can just pass the full `formatted_input` of injected as (s_e + x_e) if the attack supports it?
                        # No, the attack formulas insert things between s_e and x_e in some cases?
                        # "Naive: x_t + s_e + x_e"
                        # "Combined: ... s_e + x_e"
                        # Actually, most attacks treat (s_e + x_e) as a block or separate them.
                        # Let's check `attacks.py` implementation.
                        # Naive: `return f"{target_input} {injected_instruction} {injected_input}"`
                        
                        # So we need to extract these.
                        # We can get the template from `task_config[i_task]['prompt_template']`.
                        # But `s_e` is the instruction part.
                        
                        # Simplification:
                        # x_t: The string representation of target input.
                        # s_e: The instruction of the injected task.
                        # x_e: The input of the injected task.
                        
                        # We can get x_t from `sample['target']['original_inputs']`.
                        # We can get x_e from `sample['injected']['original_inputs']`.
                        # We can get s_e from `task_config[i_task]['prompt_template']` (minus the placeholders).
                        
                        # This is getting complicated because templates have placeholders.
                        # Alternative:
                        # The `formatted_input` in sampler is the full prompt.
                        # For target: `formatted_input` = s_t + x_t.
                        # We need to inject into x_t.
                        # So the final prompt will be s_t + (Attack(x_t, s_e, x_e)).
                        
                        # So we need:
                        # 1. s_t (Target Instruction)
                        # 2. x_t (Target Data)
                        # 3. s_e (Injected Instruction)
                        # 4. x_e (Injected Data)
                        
                        # Let's extract x_t and x_e by joining the values of `original_inputs`.
                        x_t = " ".join([str(v) for v in sample['target']['original_inputs'].values()])
                        x_e = " ".join([str(v) for v in sample['injected']['original_inputs'].values()])
                        
                        # s_t and s_e are the templates.
                        # But wait, the `formatted_input` has the filled template.
                        # If we just use the template string as s_e, it has `{text}` in it.
                        # The attack formulas in the paper usually assume s_e is the instruction text.
                        # e.g. "Ignore previous instructions" is part of the attack, but s_e is "Write a summary...".
                        
                        # Let's approximate s_e by taking the prompt template and removing placeholders?
                        # Or better, let's look at `loader.py`. It uses `prompt_template.format(**row)`.
                        # If we want s_e, we can't easily extract it from the formatted string.
                        # However, for the purpose of the attack, s_e is the "payload instruction".
                        
                        # Let's use the `formatted_input` of the injected task as (s_e + x_e) combined?
                        # Most attacks in `attacks.py` take `injected_instruction` and `injected_input`.
                        # Naive: `target_input + " " + injected_instruction + " " + injected_input`
                        # If we pass `injected_instruction` as the full formatted injected prompt and `injected_input` as empty string?
                        # Then Naive becomes: `x_t + " " + (s_e + x_e) + " " + ""`. This works.
                        # Combined: `... + " " + s_e + " " + x_e`.
                        # If we pass full formatted prompt as `s_e` and empty `x_e`, it works for all attacks that just concatenate.
                        
                        # Let's verify `CombinedAttack` in `attacks.py`.
                        # Formula: `x_t + ... + s_e + x_e`.
                        # Yes, treating `s_e` as the full injected prompt and `x_e` as empty string is mathematically equivalent for concatenation.
                        
                        injected_full_prompt = sample['injected']['formatted_input']
                        
                        # Generate adversarial input x_tilde
                        x_tilde = attack_strategy.apply(
                            target_input=x_t,
                            injected_instruction=injected_full_prompt,
                            injected_input=""
                        )
                        
                        # 2. Apply Defense
                        # The defense might modify x_tilde or flag it.
                        # Defense.apply(target_instruction, input_text, llm)
                        # We need target_instruction (s_t).
                        # We can get s_t from `task_config[t_task]['prompt_template']`.
                        # But again, templates have placeholders.
                        # Some defenses (Sandwich) need s_t.
                        # "Sandwich: s_t + x_t + ... + s_t"
                        # If we pass the raw template, it might look weird with `{text}`.
                        # But maybe that's acceptable or we should fill it with empty?
                        # Let's use the template as is.
                        
                        s_t = task_config[t_task]['prompt_template']
                        
                        final_prompt = ""
                        is_detected = False
                        
                        if defense:
                            defense_out = defense.apply(
                                target_instruction=s_t,
                                input_text=x_tilde,
                                llm=llm
                            )
                            is_detected = defense_out.get('is_detected', False)
                            final_prompt = defense_out.get('prompt', "")
                        else:
                            # No defense: Standard prompt construction
                            # Prompt = s_t + x_tilde
                            # But wait, `loader.py` formats the prompt.
                            # We can't easily use `loader` here because we have a modified input `x_tilde`.
                            # We should manually construct the prompt using the template.
                            # But the template expects keys like `{text}`.
                            # We assume `x_tilde` replaces the main input field.
                            # We need to know which field is the main input.
                            # `task_config[t_task]['input_fields']` has the list.
                            # Usually the first one or we join them.
                            # Let's assume we replace the `{text}` or similar placeholder with `x_tilde`.
                            # This is tricky.
                            
                            # Workaround:
                            # Most templates are like "Task... {text} ...".
                            # We can try to format the template with `x_tilde` as the value for all input fields?
                            # Or just concatenate s_t + x_tilde if we can't format.
                            # The paper says: "x_tilde = A(...)". "Prompt = s_t + x_tilde".
                            # This implies simple concatenation in the abstract.
                            # But for specific tasks (e.g. MRPC), the prompt is "Are these equivalent: {sent1} and {sent2}?".
                            # If we inject, where does it go?
                            # Usually injection replaces one of the inputs.
                            # Our `x_tilde` generation used `x_t` (joined inputs).
                            # So `x_tilde` is the full data block.
                            # So we should probably append `x_tilde` to the instruction `s_t`.
                            
                            # Let's try to use the `prompt_template` but replace the input part with `x_tilde`.
                            # If we can't, we fallback to `s_t + "\n" + x_tilde`.
                            
                            # Actually, `Defense.apply` returns the FULL prompt (instruction + data).
                            # So if defense is None, we need to construct it.
                            # Let's define a helper to construct the prompt.
                            
                            # If we look at `SandwichDefense`, it constructs `instruction + input + instruction`.
                            # So we should follow that pattern.
                            
                            # Let's just do:
                            final_prompt = f"{s_t}\n{x_tilde}"
                        
                        if is_detected:
                            defense_detected_count += 1
                            # If detected, what is the output?
                            # Usually defenses block execution.
                            # We can treat output as "I cannot answer" or empty.
                            # This counts as failure for Attack (Good for defense).
                            model_response = "Malicious input detected."
                        else:
                            # Query LLM
                            model_response = llm.query(final_prompt)
                        
                        # 3. Calculate Metrics
                        
                        # ASV: Match(model_response, injected_label)
                        # We need the ground truth of the INJECTED task.
                        # `sample['injected']['label']`
                        inj_label = sample['injected']['label']
                        inj_task_type = task_config[i_task]['task_type']
                        
                        asv_val = metrics_calc.calculate(
                            predictions=[model_response],
                            references=[inj_label],
                            task_type=inj_task_type
                        )
                        asv_scores.append(asv_val)
                        
                        # MR: Match(model_response, baseline_response)
                        baseline_resp = injected_baselines[idx]
                        mr_val = metrics_calc.calculate(
                            predictions=[model_response],
                            references=[baseline_resp],
                            task_type=inj_task_type # Use same metric logic (accuracy/rouge)
                        )
                        mr_scores.append(mr_val)
                        
                        # Save sample result
                        batch_results['samples'].append({
                            "id": idx,
                            "x_tilde": x_tilde,
                            "prompt": final_prompt,
                            "response": model_response,
                            "injected_label": inj_label,
                            "baseline_response": baseline_resp,
                            "asv": asv_val,
                            "mr": mr_val,
                            "detected": is_detected
                        })
                    
                    # Aggregate results for this combination
                    avg_asv = sum(asv_scores) / len(asv_scores) if asv_scores else 0.0
                    avg_mr = sum(mr_scores) / len(mr_scores) if mr_scores else 0.0
                    detection_rate = defense_detected_count / len(samples) if samples else 0.0
                    
                    batch_results['metrics'] = {
                        "asv": avg_asv,
                        "mr": avg_mr,
                        "detection_rate": detection_rate
                    }
                    
                    results.append(batch_results)
                    
                    # Save intermediate results
                    with open(os.path.join(args.output_dir, "benchmark_results_partial.json"), 'w') as f:
                        json.dump(results, f, indent=2)
                        
                    pbar.update(1)
    
    pbar.close()
    
    # Save final results
    final_path = os.path.join(args.output_dir, f"benchmark_results_{int(time.time())}.json")
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark complete. Results saved to {final_path}")

if __name__ == "__main__":
    main()
