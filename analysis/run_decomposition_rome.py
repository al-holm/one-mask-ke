import torch
import numpy as np
from tqdm import tqdm
from revit.models.model_enum import ModelName

def decompose_residual_stream(
    model, hf_model, tokenizer, prompt, token_id, W_U, device
):
    """
    Decompose residual stream into MLP and Attention contributions per layer.
    Returns the contribution of each component to the target token's logit.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if isinstance(token_id, list):
        token_id = token_id[0]

    unembed_vec = W_U[token_id]

    mlp_contributions = []
    attn_contributions = []

    mlp_outputs = {}
    attn_outputs = {}

    def make_mlp_hook(layer_idx):
        def hook(module, input, output):
            mlp_outputs[layer_idx] = output[0, -1, :].detach()  # last token

        return hook

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_outputs[layer_idx] = output[0][0, -1, :].detach()  # last token
            else:
                attn_outputs[layer_idx] = output[0, -1, :].detach()  # last token

        return hook

    hooks = []
    if model.model_name == ModelName.LLAMA3_3B:
        layers = hf_model.model.layers[1:-1]
        mlp_attr = "mlp"
        attn_attr = "self_attn"
    else:  # GPT-2
        layers = hf_model.transformer.h[1:-1]
        mlp_attr = "mlp"
        attn_attr = "attn"

    for layer_idx, layer in enumerate(layers):
        mlp_module = getattr(layer, mlp_attr)
        attn_module = getattr(layer, attn_attr)
        hooks.append(mlp_module.register_forward_hook(make_mlp_hook(layer_idx)))
        hooks.append(attn_module.register_forward_hook(make_attn_hook(layer_idx)))

    # Forward pass
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute contributions
    for layer_idx in range(len(layers)):
        mlp_contrib = torch.dot(mlp_outputs[layer_idx], unembed_vec).item()
        attn_contrib = torch.dot(attn_outputs[layer_idx], unembed_vec).item()
        mlp_contributions.append(mlp_contrib)
        attn_contributions.append(attn_contrib)

    return mlp_contributions, attn_contributions


def collect_decomposition_results(model, loader, dataset, num_cases, global_mask=None):
    """
    Collect decomposition results for original, edited, and optionally pruned models.
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
    else:
        results["edit_model_orig_fact_mlp"] = []
        results["edit_model_orig_fact_attn"] = []
        results["orig_model_edit_fact_mlp"] = []
        results["orig_model_edit_fact_attn"] = []

    for i in tqdm(range(min(num_cases, len(dataset)))):
        example = dataset[i]
        case_id = example["case_id"]
        req = example["requested_rewrite"]
        rel = req["relation_id"]

        prompt = req["prompt"].format(req["subject"])
        obj_true = req["target_true"]["str"]
        obj_new = req["target_new"]["str"]

        id_true = tokenizer.encode(obj_true, add_special_tokens=False)[0]
        id_new = tokenizer.encode(obj_new, add_special_tokens=False)[0]

        try:
            W_edited = loader.reconstruct_on_device(case_id, rel)
        except Exception:
            print(f"File not found for case {case_id}, relation {rel}")
            continue

        # 1. Original model, original fact
        model.set_weight(model.original_weight)
        mlp_c, attn_c = decompose_residual_stream(
            model, hf_model, tokenizer, prompt, id_true, W_U, device
        )
        results["orig_model_orig_fact_mlp"].append(mlp_c)
        results["orig_model_orig_fact_attn"].append(attn_c)

        # 2. Edited model, edited fact
        model.set_weight(W_edited)
        mlp_c, attn_c = decompose_residual_stream(
            model, hf_model, tokenizer, prompt, id_new, W_U, device
        )
        results["edit_model_edit_fact_mlp"].append(mlp_c)
        results["edit_model_edit_fact_attn"].append(attn_c)
        
        if global_mask is not None:
            # 5. Pruned model, original fact
            W_pruned = W_edited * global_mask.to(device, dtype=W_edited.dtype)
            model.set_weight(W_pruned)
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

            # 4. Original model, edited fact
            model.set_weight(model.original_weight)
            mlp_c, attn_c = decompose_residual_stream(
                model, hf_model, tokenizer, prompt, id_new, W_U, device
            )
            results["orig_model_edit_fact_mlp"].append(mlp_c)
            results["orig_model_edit_fact_attn"].append(attn_c)

    model.set_weight(model.original_weight)
    return results


def compute_statistics(results):
    """
    Compute mean, std, and standard error for results.
    """
    mean_results = {k: np.mean(v, axis=0) for k, v in results.items()}
    std_results = {k: np.std(v, axis=0) for k, v in results.items()}
    n = len(next(iter(results.values())))
    se_results = {k: std_results[k] / np.sqrt(n) for k in std_results}
    return mean_results, std_results, se_results, n
