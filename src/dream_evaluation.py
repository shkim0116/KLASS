import os
import json
import torch
import numpy as np
from tqdm import tqdm
import random
import re
import argparse
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from utils import *
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.dim() > 0 else obj.item()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj

def test_dataset(
    model, tokenizer, save_dir, dataset,
    gen_length, steps, 
    conf_threshold=0.9, kl_threshold=0.01, history_length=2,
    alg="klass", unmask_strategy="all",
    temperature=0.2, top_p=0.95,
    test_size=None, random_sampling=False,
    num_samples=1,
    save_steps=False
):
    if alg == "klass":
        save_dir = f"{save_dir}/Dream/{dataset}/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_s{num_samples}"
    else:
        save_dir = f"{save_dir}/Dream/{dataset}/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/s{num_samples}"
    step_save_dir = None
    if save_steps: 
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    data_path = f"./data/{dataset}_test.json"
    data = process_file(data_path)

    if test_size:
        random.seed(516)
        data = random.sample(data, test_size) if random_sampling else data[:test_size]
    
    correct_count = 0
    used_steps_list = []
    results = {"summary": {}, "results": []}

    for example_idx, example in tqdm(enumerate(data), total=len(data), desc=f"Generating completions for {dataset.capitalize()}"):
        if dataset == "gsm8k":
            prompt = example['question']
            answer = example['answer']
            ground_truth_answer = parse_ground_truth_answer(answer)
        elif dataset == "math":
            prompt = example['problem']
            solution = example['solution']
            ground_truth_answer = extract_math_answer(prompt, solution)

        example_samples = []
        example_correct = False
        example_steps = []

        messages = [
                    {"role": "system", "content": "Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format 'The final answer is'."},
                    {"role": "user", "content": prompt}
                    ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")

        for sample_idx in range(num_samples):
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                output_history=True,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                unmask_strategy=unmask_strategy,
                alg_temp=0.,
                conf_threshold=conf_threshold,
                kl_threshold=kl_threshold,
                kl_history_length=history_length,
                save_steps=save_steps
            )
            used_steps = output.used_steps
            generations = [
                tokenizer.decode(g[len(p) :].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            generated_text = generations[0].split(tokenizer.eos_token)[0]

            if dataset == "gsm8k":
                generated_answer = parse_answer(generated_text)
                is_correct = generated_answer == ground_truth_answer
            elif dataset == "math":
                generated_answer = extract_math_answer(prompt, generated_text)
                is_correct = compare_answers(prompt, ground_truth_answer, generated_answer)

            if is_correct:
                example_correct = True
            
            example_steps.append(used_steps)
            example_samples.append({
                "task_id": example_idx,
                "sample_idx": sample_idx,
                "used_steps": used_steps,
                "generation": generated_text,
                "parsed_answer": generated_answer,
                "is_correct": is_correct
            })

            if save_steps:
                history = output.history
                decoded_history = []
                for step in history:
                    text_ids = step["text"][0].tolist()
                    decoded_text = tokenizer.decode(text_ids).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " ")
                    for token in step["decoded_tokens"]:
                        token['decoded_token'] = tokenizer.decode([token['token_id']])
                    step_with_decoded = dict(step)
                    step_with_decoded["decoded_text"] = decoded_text
                    del step_with_decoded["text"]
                    step_with_decoded = tensor_to_python(step_with_decoded)
                    decoded_history.append(step_with_decoded)
                all_steps_path = os.path.join(step_save_dir, f"q{example_idx}_s{sample_idx}.json")
                with open(all_steps_path, "w") as f:
                    json.dump(decoded_history, f, indent=2)

        if example_correct:
            correct_count += 1
        
        used_steps_list.extend(example_steps)
        results["results"].append({
            "task_id": example_idx,
            "input_prompt": prompt,
            "ground_truth_answer": ground_truth_answer,
            "any_correct": example_correct,
            "avg_steps": round(sum(example_steps) / len(example_steps), 2),
            "samples": example_samples
        })
    
    accuracy = correct_count / len(data)
    avg_steps = sum(used_steps_list) / len(used_steps_list)
    results["summary"] = {
        "accuracy": round(accuracy*100, 2),
        "average_steps": round(avg_steps, 2),
        "total_questions": len(data),
        "correct_questions": correct_count,
        "num_samples_per_question": num_samples
    }

    save_path = f"{save_dir}/all_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"[{dataset.capitalize()}]")
    print(f"Accuracy: {round(accuracy*100, 2)}")
    print(f"Average steps: {round(avg_steps, 2)}")
    print(f"Results saved to {save_path}")

        
def test_humaneval(
    model, tokenizer, save_dir,
    gen_length, steps,
    conf_threshold=0.9, kl_threshold=0.001, history_length=2,
    alg="klass", unmask_strategy="all",
    temperature=0.2, top_p=0.95,
    k=(1,10,100), n_workers=4, timeout=3.0,
    test_size=None, random_sampling=False,
    num_samples=1,
    save_steps=False
):
    if alg == "klass":
        save_dir = f"{save_dir}/Dream/humaneval/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_s{num_samples}"
    else:
        save_dir = f"{save_dir}/Dream/humaneval/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/s{num_samples}"
    os.makedirs(save_dir, exist_ok=True)
    if save_steps: 
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    problems = read_problems()
    
    if test_size:
        random.seed(516)
        problems = list(problems.items())
        data = random.sample(problems, test_size) if random_sampling else problems[:test_size]
        data = dict(data)

    samples = []
    steps_per_problem = []
    i = 0
    for task_id, info in tqdm(data.items(), desc="Generating completions for HumanEval"):
        prompt = info["prompt"]
        task_samples = []
        task_steps = []

        messages = [
            {"role": "system", "content": "You complete only Python code."},
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        
        for sample_idx in range(num_samples):
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                output_history=True,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                unmask_strategy=unmask_strategy,
                alg_temp=0.,
                conf_threshold=conf_threshold,
                kl_threshold=kl_threshold,
                kl_history_length=history_length,
                save_steps=save_steps
            )
            used_steps = output.used_steps
            task_steps.append(used_steps)
            generations = [
                tokenizer.decode(g[len(p):].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            decoded_text = generations[0].split(tokenizer.eos_token)[0]

            code_match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", decoded_text, re.DOTALL)
            code_only = code_match.group(1).strip() if code_match else decoded_text.strip()

            sample_data = {
                "task_id": task_id, 
                "sample_idx": sample_idx,
                "completion": code_only, 
                "used_steps": used_steps
            }
            task_samples.append(sample_data)
            samples.append(sample_data)

            if save_steps:
                history = output.history
                decoded_history = []
                for step in history:
                    text_ids = step["text"][0].tolist()
                    decoded_step_text = tokenizer.decode(text_ids).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " ")
                    for token in step["decoded_tokens"]:
                        token['decoded_token'] = tokenizer.decode([token['token_id']])
                    step_with_decoded = dict(step)
                    step_with_decoded["decoded_text"] = decoded_step_text
                    del step_with_decoded["text"]
                    step_with_decoded = tensor_to_python(step_with_decoded)
                    decoded_history.append(step_with_decoded)
                all_steps_path = os.path.join(step_save_dir, f"q{i}_s{sample_idx}.json")
                with open(all_steps_path, "w") as f:
                    json.dump(decoded_history, f, indent=2)

        steps_per_problem.append({
            "task_id": task_id, 
            "input_prompt": prompt,
            "avg_steps": sum(task_steps) / len(task_steps),
            "samples": task_samples
        })
        i += 1

    samples_file = os.path.join(save_dir, "humaneval_samples.jsonl")
    write_jsonl(samples_file, samples)

    results = evaluate_functional_correctness(
        samples_file,
        k=",".join(map(str, k)),
        n_workers=n_workers,
        timeout=timeout
    )

    avg_steps = sum([entry["avg_steps"] for entry in steps_per_problem]) / len(steps_per_problem)

    all_results = {}
    all_results["summary"] = {
        "accuracy": round(results['pass@1']*100, 2),
        "average_steps": round(avg_steps, 2)
    }
    all_results["results"] = steps_per_problem

    save_path = os.path.join(save_dir, "all_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print('[HumanEval]')
    print("Accuracy:", round(results['pass@1']*100, 2))
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Results saved to {save_path}")


def test_mbpp(
    model, tokenizer, save_dir,
    gen_length, steps,
    conf_threshold=0.9, kl_threshold=0.001, history_length=2,
    alg="klass", unmask_strategy="all",
    temperature=0.2, top_p=0.95,
    eval_timeout=3.0,
    test_size=None, random_sampling=False,
    num_samples=1,
    save_steps=False
):
    if alg == "klass":
        save_dir = f"{save_dir}/Dream/mbpp/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_s{num_samples}"
    else:
        save_dir = f"{save_dir}/Dream/mbpp/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/s{num_samples}"
    os.makedirs(save_dir, exist_ok=True)

    if save_steps: 
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    if test_size:
        random.seed(516)
        ds = random.sample(list(ds), test_size) if random_sampling else ds.select(range(test_size))

    steps_per_problem = []

    for idx, ex in enumerate(tqdm(ds, desc="Generating completions for MBPP")):
        task_id = ex["task_id"]
        prompt  = ex["prompt"]
        tests   = ex["test_list"]
        code    = ex["code"]
        
        task_samples = []
        task_steps = []
        task_passed = False

        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]"}
            ],
            return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")
        
        for sample_idx in range(num_samples):
            
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                unmask_strategy=unmask_strategy,
                conf_threshold=conf_threshold,
                kl_threshold=kl_threshold,
                kl_history_length=history_length,
                output_history=True,
                return_dict_in_generate=True,
                save_steps=save_steps
            )
            used_steps = output.used_steps
            task_steps.append(used_steps)
            generations = [
                tokenizer.decode(g[len(p) :].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            decoded_text = generations[0].split(tokenizer.eos_token)[0]
            
            sample_data = {
                "task_id": task_id,
                "sample_idx": sample_idx,
                "used_steps": used_steps,
                "generation": decoded_text
            }
            
            # Save stepwise info if requested
            if save_steps:
                history = output.history
                decoded_history = []
                for step in history:
                    text_ids = step["text"][0].tolist()
                    decoded_step_text = tokenizer.decode(text_ids).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " ")
                    for token in step["decoded_tokens"]:
                        token['decoded_token'] = tokenizer.decode([token['token_id']])
                    step_with_decoded = dict(step)
                    step_with_decoded["decoded_text"] = decoded_step_text
                    del step_with_decoded["text"]
                    step_with_decoded = tensor_to_python(step_with_decoded)
                    decoded_history.append(step_with_decoded)
                all_steps_path = os.path.join(step_save_dir, f"q{idx}_s{sample_idx}.json")
                with open(all_steps_path, "w") as f:
                    json.dump(decoded_history, f, indent=2)
            
            # Evaluate this sample
            passed = evaluate_task(sample_data, tests, timeout=eval_timeout)
            if passed:
                task_passed = True
            sample_data["passed"] = passed
            task_samples.append(sample_data)
        
        # Aggregate results for this task
        steps_per_problem.append({
            "task_id": task_id, 
            "input_prompt": prompt,
            "solution_code": code,
            "any_passed": task_passed,
            "avg_steps": sum(task_steps) / len(task_steps),
            "samples": task_samples
        })
    
    total_passed_tasks = sum(1 for task_data in steps_per_problem if task_data["any_passed"])
    accuracy = total_passed_tasks / len(steps_per_problem) if steps_per_problem else 0
    average_steps = sum([entry["avg_steps"] for entry in steps_per_problem]) / len(steps_per_problem)

    all_results = {}
    all_results["summary"] = {
        "accuracy": round(accuracy*100, 2),
        "average_steps": round(average_steps, 2),
        "total_tasks": len(steps_per_problem),
        "passed_tasks": total_passed_tasks,
        "num_samples_per_task": num_samples
    }
    all_results["results"] = steps_per_problem

    samples_file = os.path.join(save_dir, "all_results.json")
    with open(samples_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"[MBPP]")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average steps: {average_steps:.2f}")
    print(f"Results saved to {samples_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dream model on math or GSM8K dataset.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, choices=['gsm8k', 'math', 'humaneval', 'mbpp'], help='Dataset to use')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--gen_length', type=int, default=256, help='Generation length')
    parser.add_argument('--steps', type=int, default=256, help='Number of steps')
    parser.add_argument('--conf_threshold', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--kl_threshold', type=float, default=0.01, help='KL threshold')
    parser.add_argument('--history_length', type=int, default=2, help='History length')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    parser.add_argument('--test_size', type=int, default=None, help='Number of test examples')
    parser.add_argument('--random_sampling', action='store_true', help='Use random sampling')
    parser.add_argument('--alg', type=str, choices=['klass', 'maskgit_plus', 'origin', 'entropy'], default='maskgit_plus', help='Diffusion algorithm to use')
    parser.add_argument('--unmask_strategy', type=str, choices=['all', 'max_conf', 'min_kl', 'random'], default='all', help='Unmasking strategy')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per question/task')
    parser.add_argument('--save_steps', action='store_true', help='If set, save the results of each step.')
    # HumanEval-specific arguments
    parser.add_argument('--humaneval_k', type=str, default="1", help='pass@k values, comma-separated')
    parser.add_argument('--humaneval_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--humaneval_timeout', type=float, default=3.0, help='Test timeout (seconds)')
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = model.to("cuda").eval()

    if args.dataset == 'humaneval':
        ks = tuple(map(int, args.humaneval_k.split(',')))
        test_humaneval(
            model, tokenizer, args.save_dir,
            num_samples=args.num_samples,
            k=ks,
            n_workers=args.humaneval_workers,
            timeout=args.humaneval_timeout,
            gen_length=args.gen_length,
            steps=args.steps,
            conf_threshold=args.conf_threshold,
            kl_threshold=args.kl_threshold,
            history_length=args.history_length,
            alg=args.alg,
            unmask_strategy=args.unmask_strategy,
            top_p=args.top_p,
            temperature=args.temperature,
            save_steps=args.save_steps,
            test_size=args.test_size
        )
    elif args.dataset == 'mbpp':
        test_mbpp(
            model, tokenizer, args.save_dir,
            gen_length=args.gen_length,
            steps=args.steps,
            alg=args.alg,
            unmask_strategy=args.unmask_strategy,
            conf_threshold=args.conf_threshold,
            kl_threshold=args.kl_threshold,
            temperature=args.temperature,
            test_size=args.test_size,
            top_p=args.top_p,
            save_steps=args.save_steps,
            num_samples=args.num_samples
        )
    else:
        test_dataset(
            model, tokenizer, 
            save_dir=args.save_dir,
            dataset=args.dataset,
            gen_length=args.gen_length, 
            steps=args.steps,
            conf_threshold=args.conf_threshold, 
            kl_threshold=args.kl_threshold, 
            history_length=args.history_length,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.alg, 
            unmask_strategy=args.unmask_strategy,
            test_size=args.test_size, 
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps
        )

if __name__ == "__main__":
    main()