import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_residual_stream import get_dataset
from run_decomposition_rome import (
    collect_decomposition_results,
    compute_statistics,
)

from revit import GPT2XLModel, LLamaModel, ModelName
from revit.mask_trainer import FP16DeltaLoader

random.seed(42)
FONT_SIZE_TITLE = 25
FONT_SIZE_LABEL = 19
FONT_SIZE_LEGEND = 15
FONT_SIZE_TICKS = 17


def run_decomposition_analysis(
    model, loader, dataset, global_mask, target_layer_idx=17, num_cases=50
):
    """
    Compare residual stream decomposition between original, edited, and pruned models.
    """
    target_layer_idx -= 1

    results = collect_decomposition_results(
        model, loader, dataset, num_cases, global_mask=global_mask
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
    ax1.axvline(x=target_layer_idx, color="k", linestyle=":", label="Edit Layer")
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
    ax2.axvline(x=target_layer_idx, color="k", linestyle=":", label="Edit Layer")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Contribution to Token Logit", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Attention Contributions", fontsize=FONT_SIZE_TITLE)
    ax2.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"decomposition_analysis_pruned_{model.model_name}.pdf")
    plt.show()

    return mean_results


if __name__ == "__main__":
    models = [LLamaModel(), GPT2XLModel()]
    layers = [5, 17]
    for model, layer in zip(models, layers):
        loader = FP16DeltaLoader(model.device, model_name=model.model_name)

        dataset = get_dataset()

        mask_dict = torch.load(
            f"../output/masks_shaorange/{model.model_name}/all_kl_mask_.pt"
        )
        threshold = 0.9 if model.model_name == ModelName.LLAMA3_3B else 0.7
        mask = mask_dict["mask"] > threshold
        mask = mask.to(dtype=torch.int)

        run_decomposition_analysis(
            model,
            loader,
            dataset,
            mask,
            target_layer_idx=model.target_layer,
            num_cases=1000,
        )
