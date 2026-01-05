from tqdm import tqdm
from run_decomposition_rome import decompose_residual_stream

def collect_decomposition_results_memit(model, memit_weights, dataset, num_cases, global_mask=None):
    """
    Collect decomposition results for original, edited, and optionally pruned MEMIT models.
    """
    hf_model = model.llm
    hf_model.eval()
    tokenizer = model.tokenizer
    device = model.device

    if hasattr(hf_model, "lm_head"):
        W_U = hf_model.lm_head.weight.detach().to(device)
    else:
        W_U = hf_model.get_output_embeddings().weight.detach().to(device)

    # Storage for aggregation
    results = {
        "orig_model_orig_fact_mlp": [],
        "orig_model_orig_fact_attn": [],
        "edit_model_edit_fact_mlp": [],
        "edit_model_edit_fact_attn": [],
    }
    
    if global_mask is not None:
        results["pruned_model_orig_fact_mlp"] = []
        results["pruned_model_orig_fact_attn"] = []
        target_dtype = model.original_weight.dtype
        global_mask = global_mask.to(device=device, dtype=target_dtype)
        
        # Prepare pruned weights
        pruned_memit_weights = {
            k: v * global_mask.to(v.device, dtype=v.dtype) for k, v in memit_weights.items()
        }
    else:
        results["edit_model_orig_fact_mlp"] = []
        results["edit_model_orig_fact_attn"] = []

    for i in tqdm(range(min(num_cases, len(dataset)))):
        example = dataset[i]
        req = example["requested_rewrite"]

        prompt = req["prompt"].format(req["subject"])
        obj_true = req["target_true"]["str"]
        obj_new = req["target_new"]["str"]

        id_true = tokenizer.encode(obj_true, add_special_tokens=False)[0]
        id_new = tokenizer.encode(obj_new, add_special_tokens=False)[0]

        # 1. Original model, original fact
        model.reset_weights_memit()
        mlp_c, attn_c = decompose_residual_stream(
            model, hf_model, tokenizer, prompt, id_true, W_U, device
        )
        results["orig_model_orig_fact_mlp"].append(mlp_c)
        results["orig_model_orig_fact_attn"].append(attn_c)

        # 2. Edited model, edited fact
        model.set_weights_memit(memit_weights)
        mlp_c, attn_c = decompose_residual_stream(
            model, hf_model, tokenizer, prompt, id_new, W_U, device
        )
        results["edit_model_edit_fact_mlp"].append(mlp_c)
        results["edit_model_edit_fact_attn"].append(attn_c)
        
        if global_mask is not None:
            # 3. Pruned model, original fact
            model.set_weights_memit(pruned_memit_weights)
            mlp_c, attn_c = decompose_residual_stream(
                model, hf_model, tokenizer, prompt, id_true, W_U, device
            )
            results["pruned_model_orig_fact_mlp"].append(mlp_c)
            results["pruned_model_orig_fact_attn"].append(attn_c)
        else:
            # 3. Edited model, original fact
            mlp_c, attn_c = decompose_residual_stream(
                model, hf_model, tokenizer, prompt, id_true, W_U, device
            )
            results["edit_model_orig_fact_mlp"].append(mlp_c)
            results["edit_model_orig_fact_attn"].append(attn_c)

    model.reset_weights_memit()
    return results