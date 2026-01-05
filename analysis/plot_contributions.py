import random

import matplotlib.pyplot as plt
import numpy as np
from plot_residual_stream import get_dataset
from run_decomposition_rome import (
    collect_decomposition_results,
    compute_statistics,
)

from revit import GPT2XLModel, LLamaModel
from revit.mask_trainer import FP16DeltaLoader

random.seed(42)

FONT_SIZE_TITLE = 25
FONT_SIZE_LABEL = 19
FONT_SIZE_LEGEND = 18
FONT_SIZE_TICKS = 17


def plot_decomposition_analysis(
    mean_results, std_results, se_results, n, target_layer_idx, model_name
):
    """
    Plot the decomposition analysis results.
    """
    num_layers = len(mean_results["orig_model_orig_fact_mlp"])
    layers = np.arange(num_layers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MLP contributions comparison
    ax1 = axes[0, 0]
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
        mean_results["edit_model_orig_fact_mlp"],
        "m-",
        lw=2,
        label="Original fact - Edited model",
    )
    ax1.plot(
        layers,
        mean_results["orig_model_edit_fact_mlp"],
        "c-",
        lw=2,
        label="Edited fact - Original model",
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
        mean_results["edit_model_orig_fact_mlp"]
        - se_results["edit_model_orig_fact_mlp"],
        mean_results["edit_model_orig_fact_mlp"]
        + se_results["edit_model_orig_fact_mlp"],
        color="magenta",
        alpha=0.3,
    )
    ax1.fill_between(
        layers,
        mean_results["orig_model_edit_fact_mlp"]
        - se_results["orig_model_edit_fact_mlp"],
        mean_results["orig_model_edit_fact_mlp"]
        + se_results["orig_model_edit_fact_mlp"],
        color="cyan",
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
    ax2 = axes[0, 1]
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
        mean_results["edit_model_orig_fact_attn"],
        "m-",
        lw=2,
        label="Original fact - Edited model",
    )
    ax2.plot(
        layers,
        mean_results["orig_model_edit_fact_attn"],
        "c-",
        lw=2,
        label="Edited fact - Original model",
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
        mean_results["edit_model_orig_fact_attn"]
        - se_results["edit_model_orig_fact_attn"],
        mean_results["edit_model_orig_fact_attn"]
        + se_results["edit_model_orig_fact_attn"],
        color="magenta",
        alpha=0.3,
    )
    ax2.fill_between(
        layers,
        mean_results["orig_model_edit_fact_attn"]
        - se_results["orig_model_edit_fact_attn"],
        mean_results["orig_model_edit_fact_attn"]
        + se_results["orig_model_edit_fact_attn"],
        color="cyan",
        alpha=0.3,
    )
    ax2.axvline(x=target_layer_idx, color="k", linestyle=":", label="Edit Layer")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Contribution to Token Logit", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Attention Contributions", fontsize=FONT_SIZE_TITLE)
    ax2.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax2.grid(True, alpha=0.3)

    # Plot 3: MLP comparison at edited layer (bar chart)
    ax3 = axes[1, 0]
    bar_data = [
        mean_results["orig_model_orig_fact_mlp"][target_layer_idx],
        mean_results["edit_model_edit_fact_mlp"][target_layer_idx],
        mean_results["edit_model_orig_fact_mlp"][target_layer_idx],
    ]
    ax3.bar(
        [
            "Original fact\nOriginal model",
            "Edit fact\nEdited model",
            "Original fact\nEdited model",
        ],
        bar_data,
        color=["green", "blue", "magenta"],
        yerr=[
            se_results["orig_model_orig_fact_mlp"][target_layer_idx],
            se_results["edit_model_edit_fact_mlp"][target_layer_idx],
            se_results["edit_model_orig_fact_mlp"][target_layer_idx],
        ],
    )
    ax3.set_ylabel("Contribution to Token Logit", fontsize=FONT_SIZE_LABEL)
    ax3.set_title("MLP Contribution at Edited Layer", fontsize=FONT_SIZE_TITLE)
    ax3.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    # Plot 4: Cumulative contributions
    ax4 = axes[1, 1]
    cumsum_orig = np.cumsum(mean_results["orig_model_orig_fact_mlp"]) + np.cumsum(
        mean_results["orig_model_orig_fact_attn"]
    )
    cumsum_edit = np.cumsum(mean_results["edit_model_edit_fact_mlp"]) + np.cumsum(
        mean_results["edit_model_edit_fact_attn"]
    )

    std_cum_orig = np.sqrt(
        np.cumsum(std_results["orig_model_orig_fact_mlp"] ** 2)
        + np.cumsum(std_results["orig_model_orig_fact_attn"] ** 2)
    )
    std_cum_edit = np.sqrt(
        np.cumsum(std_results["edit_model_edit_fact_mlp"] ** 2)
        + np.cumsum(std_results["edit_model_edit_fact_attn"] ** 2)
    )
    se_cum_orig = std_cum_orig / np.sqrt(n)
    se_cum_edit = std_cum_edit / np.sqrt(n)

    ax4.plot(layers, cumsum_orig, "g-", lw=2, label="Original fact (Original model)")
    ax4.plot(layers, cumsum_edit, "b-", lw=2, label="Edited fact (Edited model)")
    ax4.fill_between(
        layers,
        cumsum_orig - se_cum_orig,
        cumsum_orig + se_cum_orig,
        color="green",
        alpha=0.3,
    )
    ax4.fill_between(
        layers,
        cumsum_edit - se_cum_edit,
        cumsum_edit + se_cum_edit,
        color="blue",
        alpha=0.3,
    )
    ax4.axvline(x=target_layer_idx, color="k", linestyle=":", label="Edit Layer")
    ax4.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel("Cumulative Contribution to Logit", fontsize=FONT_SIZE_LABEL)
    ax4.set_title("Contribution Trace", fontsize=FONT_SIZE_TITLE)
    ax4.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"decomposition_analysis_{model_name}.pdf")
    plt.show()


def run_decomposition_analysis(
    model, loader, dataset, target_layer_idx=17, num_cases=50
):
    """
    Compare residual stream decomposition between original and edited models.
    """
    target_layer_idx -= 1

    results = collect_decomposition_results(model, loader, dataset, num_cases)
    mean_results, std_results, se_results, n = compute_statistics(results)

    plot_decomposition_analysis(
        mean_results, std_results, se_results, n, target_layer_idx, model.model_name
    )

    return mean_results


if __name__ == "__main__":
    models = [LLamaModel(), GPT2XLModel()]
    layers = [5, 17]
    for model, layer in zip(models, layers):
        print(f"Running analysis for {model.model_name}...")
        loader = FP16DeltaLoader(model.device, model_name=model.model_name)
        dataset = get_dataset()
        run_decomposition_analysis(
            model, loader, dataset, target_layer_idx=layer, num_cases=100
        )
