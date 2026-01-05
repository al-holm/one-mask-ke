import json

import pandas as pd
import torch
from tqdm import tqdm

from revit import TopKAccuracyEvaluator
from revit.models import GPT2XLModel, LLamaModel, ModelName

RELS = ["P103", "P17", "P495", "P176", "P413", "P136", "P30", "P937", "P27", "P1412"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYER = 5


def get_data(split: str):
    dataset_path = f"../res/dsets/memit_{split}_1000s_10rels.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


def evaluate_memit_mask(
    model,
    memit_weights=None,
    mask=None,
    threshold=None,
    eval_mode="pruned",
    split="test",
    include_kl=True,
):
    """
    eval_mode:
        - 'original': Evaluate original model (no weights set)
        - 'edited': Evaluate edited model (MEMIT weights set)
        - 'pruned': Evaluate pruned model (MEMIT weights + mask on target layer)
    """
    dataset = get_data(split)
    topK_evaluator = TopKAccuracyEvaluator(model)

    if include_kl:
        teacher = (
            GPT2XLModel() if model.model_name == ModelName.GPT2XL else LLamaModel()
        )
        teacher.llm.to(DEVICE)
        teacher.llm.eval()

    results = []

    w_pruned = 0.0
    rsr_total = 0.0
    i_processed = 0
    if eval_mode == "edited":
        print("Setting MEMIT weights for edited evaluation...")
        model.set_weights_memit(memit_weights)
    elif eval_mode == "pruned":
        print(
            f"Setting MEMIT weights and applying mask with threshold {threshold} for pruned evaluation..."
        )
        w_pruned = mask.sum().item() / mask.numel()
        model.set_weights_memit(memit_weights)

        edited_weight = model._get_target_weight(layer=LAYER)
        pruned_weight = edited_weight * mask
        model._set_weight_to_layer(pruned_weight, layer=LAYER)

    progress_bar = tqdm(range(len(dataset)), desc=f"Evaluation: {eval_mode}")
    for i in progress_bar:
        example = dataset[i]
        case_id = example["case_id"]
        req = example["requested_rewrite"]
        rel = req["relation_id"]
        prompt = req["prompt"].format(req["subject"])
        true_obj = req["target_true"]["str"]
        new_obj = req["target_new"]["str"]

        res = model.evaluate_rome_rsr(prompt, true_obj, new_obj)
        rsr = res["rsr"]
        top_k = topK_evaluator.evaluate(model, prompt, true_obj, k=10)

        if mode != "original":
            if include_kl:
                log_probs_student = model.get_next_token_probabilities(prompt)
                log_probs_teacher = teacher.get_next_token_probabilities(prompt)

                kl_val = torch.nn.functional.kl_div(
                    log_probs_student, torch.exp(log_probs_teacher), reduction="sum"
                ).item()

        rsr_total += float(rsr)
        i_processed += 1
        progress_bar.set_postfix({"Avg RSR": rsr_total / i_processed})

        results.append(
            {
                "case_id": case_id,
                "relation": rel,
                "is reversed": rsr,
                "kl": kl_val if include_kl else None,
                "model": str(model.model_name),
                "threshold": threshold,
                "weight_pruned_ratio": round(w_pruned, 3),
                "eval_mode": eval_mode,
                **top_k,
            }
        )
    return results


if __name__ == "__main__":
    split = "train"
    mask_dict = torch.load(
        "../output/masks_shared_memit/llama3-3b/kl_mask.pt",
        map_location=DEVICE,
    )
    threshold = 0.9
    mask = mask_dict["mask"] > threshold

    memit_weights = torch.load(
        f"../output/memit_w/llama3-3b_1000s_10rels_{split}_MEMIT/memit_batch.pt"
    )
    modes = ["original", "edited", "pruned"]
    for mode in modes:
        print(f"Running {mode} Evaluation...")
        all_results = []
        model = LLamaModel(target_layer=8, is_rome=False)
        all_results = evaluate_memit_mask(
            model,
            memit_weights=memit_weights,
            mask=mask,
            threshold=threshold,
            eval_mode=mode,
            split=split,
            include_kl=False,
        )

        df = pd.DataFrame(all_results)
        df.to_csv(f"memit_llama3-3b_eval_{split}_{mode}.csv")
