import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import math

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def stable_confident_decode(
    model, tokenizer, input_ids_original, gen_length, steps, block_length, temperature=0., mask_id=126336,
    conf_threshold=0.9, kl_threshold=0.01, kl_history_length=2, 
    step_save_dir=None, example_idx=0,
    alg="default",
    unmask_strategy="all"
):
    """
    Decoding strategy: Unmask tokens that are both high-confidence and have stable (low KL-divergence) softmax distributions over H steps.
    Implements alg options: default, random, topk_margin, entropy.
    """
    mask_id = 126336
    x = torch.full((1, input_ids_original.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids_original.shape[1]] = input_ids_original.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    used_steps = 0

    # History buffers
    V = model.lm_head.out_features if hasattr(model, "lm_head") else model.config.vocab_size
    kl_history = torch.zeros((1, x.shape[1], kl_history_length), dtype=torch.float64, device=x.device)
    p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64, device=x.device)

    all_step_outputs = []

    for num_block in range(num_blocks):
        block_start = input_ids_original.shape[1] + num_block * block_length
        block_end = input_ids_original.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            # --- Restrict to current block ---
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            mask_index = mask_index & block_mask

            # --- Break if all tokens in current block are unmasked ---
            if not mask_index[:, block_start:block_end].any():
                break

            logits = model(x).logits
            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)
            p_curr = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = torch.argmax(p_curr, dim=-1)

            # --- Compute confidence according to alg ---
            if alg == "random":
                curr_conf = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif alg == "topk_margin":
                sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
                top1 = sorted_probs[..., 0]
                top2 = sorted_probs[..., 1]
                curr_conf = top1 - top2
            elif alg == "entropy":
                eps_ent = 1e-10
                log_p = torch.log(p_curr + eps_ent)
                curr_conf = -torch.sum(p_curr * log_p, dim=-1)  # negative entropy (lower entropy = higher confidence)
            else:  # default (top confidence)
                curr_conf = torch.squeeze(torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            # KL divergence between current and previous step
            eps = 1e-12
            kl_current_prev = (p_curr * (torch.log(p_curr + eps)
                            - torch.log(p_prev + eps))
                 ).sum(dim=-1)
            # Shift kl_history and insert new KL at the end
            kl_history = torch.roll(kl_history, shifts=-1, dims=-1)
            kl_history[..., -1] = kl_current_prev

            p_prev = p_curr.clone()

            if alg == "klass":
                # --- KL threshold logic ---
                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)
                # --- Confidence threshold logic ---
                conf_mask = curr_conf > conf_threshold

                ready_mask = stable_mask & conf_mask & mask_index
            else:
                ready_mask = torch.zeros_like(curr_conf, dtype=torch.bool)

            # Select top-k tokens to unmask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
            decoded_token_info = [] 

            if step_save_dir:
                all_tokens_info = []
                for j in range(mask_index.shape[0]):
                    masked_indices_in_block = torch.where(mask_index[j, block_start:block_end])[0] + block_start
                    
                    for idx in masked_indices_in_block:
                        token_id = x0[j, idx].item()
                        # decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
                        conf_val = curr_conf[j, idx].item()
                        all_tokens_info.append({
                            "position": int(idx),
                            "token_id": token_id,
                            # "decoded_token": decoded_token,
                            "confidence": round(float(conf_val), 4),
                            "kl_divergence": "inf" if math.isinf(kl_current_prev[j, idx]) else round(float(kl_current_prev[j, idx]), 6)
                        })

            for j in range(ready_mask.shape[0]):
                ready_indices = torch.where(ready_mask[j])[0]
                if len(ready_indices) > 0:
                    if len(ready_indices) > 1 and unmask_strategy != "all":
                        if unmask_strategy == "max_conf":
                            # Pick the one with highest confidence
                            conf_vals = curr_conf[j, ready_indices]
                            max_idx = torch.argmax(conf_vals)
                            selected_indices = ready_indices[max_idx:max_idx+1]
                        elif unmask_strategy == "min_kl":
                            # Pick the one with lowest KL divergence
                            kl_vals = kl_current_prev[j, ready_indices]
                            min_idx = torch.argmin(kl_vals)
                            selected_indices = ready_indices[min_idx:min_idx+1]
                        elif unmask_strategy == "random":
                            selected_indices = ready_indices[torch.randint(0, len(ready_indices), (1,))]
                        else:
                            selected_indices = ready_indices
                    else:
                        selected_indices = ready_indices
                    transfer_index[j, selected_indices] = True
                # If no tokens meet both criteria, select top-k by confidence
                else:
                    curr_conf[:, input_ids_original.shape[1] + (num_block + 1) * block_length:] = -np.inf
                    confidence = torch.where(mask_index, curr_conf, -np.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    _, selected_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, step].item())
                    transfer_index[j, selected_indices] = True
                
                # Save info for each selected token
                if step_save_dir:
                    for idx in selected_indices:
                        token_id = x0[j, idx].item()
                        decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
                        conf_val = curr_conf[j, idx].item()
                        decoded_token_info.append({
                            "position": int(idx),
                            "token_id": token_id,
                            "decoded_token": decoded_token,
                            "confidence": round(float(conf_val), 4),
                            "kl_divergence": "inf" if math.isinf(kl_current_prev[j, idx]) else round(float(kl_current_prev[j, idx]), 6),
                        })

            x[transfer_index] = x0[transfer_index]

            if step_save_dir:
                decoded_text = tokenizer.batch_decode(x[:, input_ids_original.shape[1]:], skip_special_tokens=True)[0]
                step_out = {
                    "step": used_steps + 1,
                    "decoded_text": decoded_text,
                    "decoded_tokens_num": len(decoded_token_info),
                    "decoded_tokens": decoded_token_info,
                    "all_tokens": all_tokens_info
                }
                all_step_outputs.append(step_out)

            used_steps += 1

    if step_save_dir:
        all_steps_path = os.path.join(step_save_dir, f"all_steps_{example_idx}.json")
        with open(all_steps_path, "w") as f:
            json.dump(all_step_outputs, f, indent=2)

    return x, used_steps