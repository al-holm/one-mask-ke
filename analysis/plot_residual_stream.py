import json
import os
import random

import matplotlib.pyplot as plt
import torch

from revit import LLamaModel
from revit.mask_trainer import FP16DeltaLoader

random.seed(42)


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


def run_dimension_tracking_simple(
    model, loader, example, global_mask, target_layer_idx=17
):
    hf_model = model.llm
    hf_model.eval()
    tokenizer = model.tokenizer
    device = model.device
    target_dtype = model.original_weight.dtype
    global_mask = global_mask.to(device=device, dtype=target_dtype)

    case_id = example["case_id"]
    req = example["requested_rewrite"]
    rel = req["relation_id"]
    prompt = req["prompt"].format(req["subject"])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    W_edited = loader.reconstruct_on_device(case_id, rel)
    W_pruned = W_edited * global_mask

    # Original model
    model.set_weight(model.original_weight)
    with torch.no_grad():
        out = hf_model(**inputs, output_hidden_states=True)
    h_orig = torch.stack([h[0, -1, :] for h in out.hidden_states[1:]]).cpu().numpy()

    # Edited model
    model.set_weight(W_edited)
    with torch.no_grad():
        out = hf_model(**inputs, output_hidden_states=True)
    h_edit = torch.stack([h[0, -1, :] for h in out.hidden_states[1:]]).cpu().numpy()

    # Pruned model
    model.set_weight(W_pruned)
    with torch.no_grad():
        out = hf_model(**inputs, output_hidden_states=True)
    h_prune = torch.stack([h[0, -1, :] for h in out.hidden_states[1:]]).cpu().numpy()

    model.set_weight(model.original_weight)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin = -1
    vmax = 1
    ax1 = axes[0]
    im1 = ax1.imshow(h_orig.T, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Layer", fontsize=24)
    ax1.set_ylabel("Hidden Dimension", fontsize=24)
    # ax1.set_title("Original Model", fontsize=18)
    ax1.axvline(x=target_layer_idx, color="yellow", linestyle="--", lw=2)
    ax1.tick_params(axis="x", labelsize=24)
    # plt.colorbar(im1, ax=ax1)
    ax2 = axes[1]
    im2 = ax2.imshow(h_edit.T, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax2.set_xlabel("Layer", fontsize=24)
    ax2.set_ylabel("Hidden Dimension", fontsize=24)
    # ax2.set_title("Edited Model", fontsize=18)
    ax2.axvline(x=target_layer_idx, color="yellow", linestyle="--", lw=2)
    ax2.tick_params(axis="x", labelsize=24)
    # plt.colorbar(im2, ax=ax2)
    ax3 = axes[2]
    im3 = ax3.imshow(h_prune.T, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax3.set_xlabel("Layer", fontsize=24)
    ax3.set_ylabel("Hidden Dimension", fontsize=24)
    # ax3.set_title("Pruned Model", fontsize=18)
    ax3.axvline(x=target_layer_idx, color="yellow", linestyle="--", lw=2)
    ax3.tick_params(axis="x", labelsize=24)
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig("residual_stream_comparison.pdf")
    plt.show()


if __name__ == "__main__":
    model = LLamaModel()
    loader = FP16DeltaLoader(model.device, model_name=model.model_name)

    dataset = get_dataset()

    mask_dict = torch.load(f"../output/masks_shared/{model.model_name}/all_kl_mask_.pt")
    mask = (
        mask_dict["mask"] > 0.9
        if model.model_name == "llama3-3b"
        else mask_dict["mask"] > 0.7
    )
    mask = mask.to(dtype=torch.int)

    run_dimension_tracking_simple(model, loader, dataset[3], mask)
