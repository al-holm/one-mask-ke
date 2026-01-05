import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from run_decomposition_memit import (
    collect_decomposition_results_memit,
)
from run_decomposition_rome import (compute_statistics)
from plot_contributions_memit import get_dataset

from revit import GPT2XLModel, LLamaModel, ModelName

random.seed(42)
FONT_SIZE_TITLE = 25
FONT_SIZE_LABEL = 19
FONT_SIZE_LEGEND = 15
FONT_SIZE_TICKS = 17


def run_decomposition_analysis(
    model, memit_weights, dataset, global_mask, target_layer_idx=17, num_cases=50
):
    """
    Compare residual stream decomposition between original, edited, and pruned models.
    """
    target_layer_idx -= 1

    results = collect_decomposition_results_memit(
        model, memit_weights, dataset, num_cases, global_mask=global_mask
    )
    mean_results, std_results, se_results, n = compute_statistics(results)

    # Plot
    num_layers = len(mean_results["orig_model_orig_fact_mlp"])
    layers = np.arange(num_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: MLP contributions comparison
    ax1 = axes[0]
    ax1.plot(
        layers,
        mean_results["orig_model_orig_fact_mlp"],
        "g-",
        lw=2,
        label="Original fact - Original model",
    )
    ax1.plot(
        layers,
        mean_results["edit_model_edit_fact_mlp"],
        "b-",
        lw=2,
        label="Edited fact - Edited model",
    )
    ax1.plot(
        layers,
        mean_results["pruned_model_orig_fact_mlp"],
        "orange",
        lw=2,
        label="Original fact - Pruned model",
    )
    ax1.fill_between(
        layers,
        mean_results["orig_model_orig_fact_mlp"]
        - se_results["orig_model_orig_fact_mlp"],
        mean_results["orig_model_orig_fact_mlp"]
        + se_results["orig_model_orig_fact_mlp"],
        color="green",
        alpha=0.3,
    )
    ax1.fill_between(
        layers,
        mean_results["edit_model_edit_fact_mlp"]
        - se_results["edit_model_edit_fact_mlp"],
        mean_results["edit_model_edit_fact_mlp"]
        + se_results["edit_model_edit_fact_mlp"],
        color="blue",
        alpha=0.3,
    )
    ax1.fill_between(
        layers,
        mean_results["pruned_model_orig_fact_mlp"]
        - se_results["pruned_model_orig_fact_mlp"],
        mean_results["pruned_model_orig_fact_mlp"]
        + se_results["pruned_model_orig_fact_mlp"],
        color="orange",
        alpha=0.3,
    )
    ax1.axvline(x=target_layer_idx, color="k", linestyle=":", label="Last Edit Layer")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Contribution to Token Logit", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("MLP Contributions", fontsize=FONT_SIZE_TITLE)
    ax1.legend(loc="lower left", fontsize=FONT_SIZE_LEGEND)
    ax1.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Attention contributions comparison
    ax2 = axes[1]
    ax2.plot(
        layers,
        mean_results["orig_model_orig_fact_attn"],
        "g-",
        lw=2,
        label="Original fact - Original model",
    )
    ax2.plot(
        layers,
        mean_results["edit_model_edit_fact_attn"],
        "b-",
        lw=2,
        label="Edited fact - Edited model",
    )
    ax2.plot(
        layers,
        mean_results["pruned_model_orig_fact_attn"],
        "orange",
        lw=2,
        label="Original fact - Pruned model",
    )
    ax2.fill_between(
        layers,
        mean_results["orig_model_orig_fact_attn"]
        - se_results["orig_model_orig_fact_attn"],
        mean_results["orig_model_orig_fact_attn"]
        + se_results["orig_model_orig_fact_attn"],
        color="green",
        alpha=0.3,
    )
    ax2.fill_between(
        layers,
        mean_results["edit_model_edit_fact_attn"]
        - se_results["edit_model_edit_fact_attn"],
        mean_results["edit_model_edit_fact_attn"]
        + se_results["edit_model_edit_fact_attn"],
        color="blue",
        alpha=0.3,
    )
    ax2.fill_between(
        layers,
        mean_results["pruned_model_orig_fact_attn"]
        - se_results["pruned_model_orig_fact_attn"],
        mean_results["pruned_model_orig_fact_attn"]
        + se_results["pruned_model_orig_fact_attn"],
        color="orange",
        alpha=0.3,
    )
    ax2.axvline(x=target_layer_idx, color="k", linestyle=":", label="Last Edit Layer")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Contribution to Token Logit", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Attention Contributions", fontsize=FONT_SIZE_TITLE)
    ax2.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"decomposition_analysis_memit_pruned_{model.model_name}.pdf")
    plt.show()

    return mean_results


if __name__ == "__main__":
    models = [LLamaModel(is_rome=False), GPT2XLModel(is_rome=False)]

    # Paths for MEMIT weights
    memit_weight_paths = {
        ModelName.LLAMA3_3B: "../output/memit_w/llama3-3b_1000s_10rels_test_MEMIT/memit_batch.pt",
        ModelName.GPT2XL: "../output/memit_w/gpt2-xl_1000s_10rels_test_MEMIT/memit_batch.pt",
    }

    # Paths for MEMIT masks
    mask_paths = {
        ModelName.LLAMA3_3B: "../output/masks_shared_memit/llama3-3b/kl_mask_20260102_213632.pt",
        ModelName.GPT2XL: "../output/masks_shared_memit/gpt2-xl/kl_mask.pt",
    }

    thresholds = {ModelName.LLAMA3_3B: 0.9, ModelName.GPT2XL: 0.7}
    target_layer_idxs = {ModelName.LLAMA3_3B: 8, ModelName.GPT2XL: 17}

    for model in models:
        if model.model_name not in memit_weight_paths or not os.path.exists(
            memit_weight_paths[model.model_name]
        ):
            print(f"Skipping {model.model_name} as weights not found.")
            continue

        memit_weights = torch.load(memit_weight_paths[model.model_name])
        dataset = get_dataset()

        mask_dict = torch.load(mask_paths[model.model_name])
        threshold = thresholds[model.model_name]
        mask = mask_dict["mask"] > threshold
        mask = mask.to(dtype=torch.int)

        run_decomposition_analysis(
            model,
            memit_weights,
            dataset,
            mask,
            target_layer_idx=target_layer_idxs[model.model_name],
            num_cases=1000,
        )
