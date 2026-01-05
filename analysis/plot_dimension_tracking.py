import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_residual_stream import get_dataset
from tqdm import tqdm

from revit import GPT2XLModel, LLamaModel
from revit.mask_trainer import FP16DeltaLoader

random.seed(42)


def plot_dimension_tracking_top5(
    model, loader, dataset, global_mask, target_layer_idx=None, num_cases=50
):
    """Track top 5 pruned dimensions"""
    if target_layer_idx is None:
        target_layer_idx = model.target_layer

    hf_model = model.llm
    hf_model.eval()
    tokenizer = model.tokenizer
    device = model.device
    target_dtype = model.original_weight.dtype
    global_mask = global_mask.to(device=device, dtype=target_dtype)

    hidden_size = hf_model.config.hidden_size
    # Determine which dimension of the mask corresponds to the hidden size
    if global_mask.shape[1] == hidden_size:
        # GPT-2 style: (intermediate, hidden)
        dim_to_sum = 0
    elif global_mask.shape[0] == hidden_size:
        # Llama style: (hidden, intermediate)
        dim_to_sum = 1
    else:
        print(
            f"Warning: Could not match mask shape {global_mask.shape} to hidden size {hidden_size}. Using default dim=0."
        )
        dim_to_sum = 0

    # Find top 5 pruned dimensions
    pruning_scores = (1 - global_mask).sum(dim=dim_to_sum).cpu().numpy()
    top_dims = np.argsort(pruning_scores)[-5:][::-1]

    total_weights_per_dim = global_mask.shape[dim_to_sum]

    print("Top 5 pruned dimensions:")
    for dim in top_dims:
        pruned_count = pruning_scores[dim]
        pct = 100 * pruned_count / total_weights_per_dim
        print(
            f"  Dim {dim}: {pruned_count:.0f}/{total_weights_per_dim} pruned ({pct:.1f}%)"
        )
    storage = {dim: {"orig": [], "edit": [], "prune": []} for dim in top_dims}

    for i in tqdm(range(min(num_cases, len(dataset)))):
        example = dataset[i]
        case_id = example["case_id"]
        req = example["requested_rewrite"]
        rel = req["relation_id"]
        prompt = req["prompt"].format(req["subject"])

        W_edited = loader.reconstruct_on_device(case_id, rel)
        W_pruned = W_edited * global_mask
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        for W, key in [
            (model.original_weight, "orig"),
            (W_edited, "edit"),
            (W_pruned, "prune"),
        ]:
            model.set_weight(W)
            with torch.no_grad():
                out = hf_model(**inputs, output_hidden_states=True)
            for dim in top_dims:
                vals = [h[0, -1, dim].item() for h in out.hidden_states[1:]]
                storage[dim][key].append(vals)

    model.set_weight(model.original_weight)

    # Compute means and standard errors
    n = len(storage[top_dims[0]]["orig"])
    means = {}
    ses = {}
    for dim in top_dims:
        means[dim] = {}
        ses[dim] = {}
        for key in ["orig", "edit", "prune"]:
            arr = np.array(storage[dim][key])
            means[dim][key] = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            ses[dim][key] = std / np.sqrt(n)

    # Plot 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, dim in enumerate(top_dims):
        layers = np.arange(len(means[dim]["orig"]))

        pct = 100 * pruning_scores[dim] / total_weights_per_dim

        ax = axes[idx]
        ax.plot(layers, means[dim]["orig"], "g-", lw=2, label="Original")
        ax.fill_between(
            layers,
            means[dim]["orig"] - ses[dim]["orig"],
            means[dim]["orig"] + ses[dim]["orig"],
            color="green",
            alpha=0.3,
        )
        ax.plot(layers, means[dim]["edit"], "b-", lw=2, label="Edited")
        ax.fill_between(
            layers,
            means[dim]["edit"] - ses[dim]["edit"],
            means[dim]["edit"] + ses[dim]["edit"],
            color="blue",
            alpha=0.3,
        )
        ax.plot(layers, means[dim]["prune"], "r--", lw=2, label="Pruned")
        ax.fill_between(
            layers,
            means[dim]["prune"] - ses[dim]["prune"],
            means[dim]["prune"] + ses[dim]["prune"],
            color="red",
            alpha=0.3,
        )
        ax.axvline(x=target_layer_idx, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer", fontsize=19)
        ax.set_ylabel("Activation", fontsize=19)
        ax.set_title(f"Dim {dim} ({pct:.1f}% pruned)", fontsize=19)
        ax.tick_params(axis="x", labelsize=17)
        ax.tick_params(axis="y", labelsize=17)
        ax.grid(True, alpha=0.3)
        if idx == 4:
            ax.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(f"top5_pruned_dimensions_tracking_{model.model_name}.pdf")
    plt.show()


if __name__ == "__main__":
    models = [LLamaModel(), GPT2XLModel()]
    for model in models:
        loader = FP16DeltaLoader(model.device, model_name=model.model_name)

        dataset = get_dataset()

        mask_dict = torch.load(
            f"../output/masks_shared/{model.model_name}/all_kl_mask_.pt"
        )
        mask = (
            mask_dict["mask"] > 0.7
            if model.model_name == "gpt2-xl"
            else mask_dict["mask"] > 0.9
        )
        mask = mask.to(dtype=torch.int)

        plot_dimension_tracking_top5(model, loader, dataset, mask, num_cases=1000)
