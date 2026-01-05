import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from run_decomposition_memit import (
    collect_decomposition_results_memit,
)
from run_decomposition_rome import (compute_statistics)

from revit import GPT2XLModel, LLamaModel
from revit.models.model_enum import ModelName

random.seed(42)

FONT_SIZE_TITLE = 25
FONT_SIZE_LABEL = 19
FONT_SIZE_LEGEND = 15
FONT_SIZE_TICKS = 17


def get_dataset():
    rels = [
        "P103",
        "P17",
        "P495",
        "P176",
        "P413",
        "P136",
        "P30",
        "P937",
        "P27",
        "P1412",
    ]
    dataset = []
    for r in rels:
        path = f"../res/dsets/counterfact_test{r}.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            dataset.extend(data[:100])
    random.shuffle(dataset)
    return dataset


def run_decomposition_analysis(
    model, memit_weights, dataset, target_layer_idx=17, num_cases=50
):
    """
    Compare residual stream decomposition between original and edited models.
    """
    target_layer_idx -= 1

    results = collect_decomposition_results_memit(model, memit_weights, dataset, num_cases)
    mean_results, std_results, se_results, n = compute_statistics(results)

    # Plot
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
    ax1.axvline(x=target_layer_idx, color="k", linestyle=":", label="Last Edit Layer")
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
    )
    ax2.plot(
        layers,
        mean_results["edit_model_edit_fact_attn"],
        "b-",
        lw=2,
    )
    ax2.plot(
        layers,
        mean_results["edit_model_orig_fact_attn"],
        "m-",
        lw=2,
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
    ax2.axvline(x=target_layer_idx, color="k", linestyle=":", label="Last Edit Layer")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Contribution to Token Logit", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Attention Contributions", fontsize=FONT_SIZE_TITLE)
    ax2.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax2.grid(True, alpha=0.3)

    # Plot 3: MLP contributions across layers
    if model.model_name == ModelName.GPT2XL:
        edited_layers = [12, 13, 14, 15, 16]
    else:
        edited_layers = [3, 4, 5, 6, 7]
    ax3 = axes[1, 0]
    ax3.plot(
        [x + 1 for x in edited_layers],
        mean_results["orig_model_orig_fact_mlp"][edited_layers],
        "g-",
        lw=2,
    )
    ax3.plot(
        [x + 1 for x in edited_layers],
        mean_results["edit_model_edit_fact_mlp"][edited_layers],
        "b-",
        lw=2,
    )
    ax3.plot(
        [x + 1 for x in edited_layers],
        mean_results["edit_model_orig_fact_mlp"][edited_layers],
        "m-",
        lw=2,
    )
    edited_layer_indices = np.array(edited_layers)
    ax3.fill_between(
        [x + 1 for x in edited_layers],
        mean_results["orig_model_orig_fact_mlp"][edited_layers]
        - se_results["orig_model_orig_fact_mlp"][edited_layers],
        mean_results["orig_model_orig_fact_mlp"][edited_layers]
        + se_results["orig_model_orig_fact_mlp"][edited_layers],
        color="green",
        alpha=0.3,
    )
    ax3.fill_between(
        [x + 1 for x in edited_layers],
        mean_results["edit_model_edit_fact_mlp"][edited_layers]
        - se_results["edit_model_edit_fact_mlp"][edited_layers],
        mean_results["edit_model_edit_fact_mlp"][edited_layers]
        + se_results["edit_model_edit_fact_mlp"][edited_layers],
        color="blue",
        alpha=0.3,
    )
    ax3.fill_between(
        [x + 1 for x in edited_layers],
        mean_results["edit_model_orig_fact_mlp"][edited_layers]
        - se_results["edit_model_orig_fact_mlp"][edited_layers],
        mean_results["edit_model_orig_fact_mlp"][edited_layers]
        + se_results["edit_model_orig_fact_mlp"][edited_layers],
        color="magenta",
        alpha=0.3,
    )
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax3.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax3.set_xticks([x + 1 for x in edited_layers])
    ax3.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax3.set_ylabel("Contribution to Token Logit", fontsize=FONT_SIZE_LABEL)
    ax3.set_title("MLP Contributions Across Edited Layers", fontsize=FONT_SIZE_TITLE)
    ax3.legend(fontsize=FONT_SIZE_LEGEND)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative contributions
    ax4 = axes[1, 1]
    cumsum_orig = np.cumsum(mean_results["orig_model_orig_fact_mlp"]) + np.cumsum(
        mean_results["orig_model_orig_fact_attn"]
    )
    cumsum_edit = np.cumsum(mean_results["edit_model_edit_fact_mlp"]) + np.cumsum(
        mean_results["edit_model_edit_fact_attn"]
    )
    std_cum_orig = np.sqrt(
        np.cumsum(se_results["orig_model_orig_fact_mlp"] ** 2)
        + np.cumsum(se_results["orig_model_orig_fact_attn"] ** 2)
    )
    std_cum_edit = np.sqrt(
        np.cumsum(se_results["edit_model_edit_fact_mlp"] ** 2)
        + np.cumsum(se_results["edit_model_edit_fact_attn"] ** 2)
    )
    ax4.plot(layers, cumsum_orig, "g-", lw=2, label="Original fact (Original model)")
    ax4.plot(layers, cumsum_edit, "b-", lw=2, label="Edited fact (Edited model)")
    ax4.fill_between(
        layers,
        cumsum_orig - std_cum_orig,
        cumsum_orig + std_cum_orig,
        color="green",
        alpha=0.3,
    )
    ax4.fill_between(
        layers,
        cumsum_edit - std_cum_edit,
        cumsum_edit + std_cum_edit,
        color="blue",
        alpha=0.3,
    )
    ax4.axvline(x=target_layer_idx, color="k", linestyle=":", label="Edited Layer")
    ax4.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel("Cumulative Contribution to Logit", fontsize=FONT_SIZE_LABEL)
    ax4.set_title("Contribution Trace", fontsize=FONT_SIZE_TITLE)
    ax4.tick_params(axis="x", labelsize=FONT_SIZE_TICKS)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("decompostion_analysis_memit.pdf")
    plt.show()

    return mean_results


if __name__ == "__main__":
    models = [GPT2XLModel(is_rome=False), LLamaModel(is_rome=False)]
    for model in models:
        memit_weights = torch.load(
            f"../output/memit_w/{model.model_name}_1000s_10rels_test_MEMIT/memit_batch.pt"
        )

        dataset = get_dataset()
        target_layer_idx = 17 if model.model_name == ModelName.GPT2XL else 5

        run_decomposition_analysis(
            model,
            memit_weights,
            dataset,
            target_layer_idx=target_layer_idx,
            num_cases=1000,
        )
