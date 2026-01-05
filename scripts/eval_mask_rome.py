import pandas as pd
import torch
from tqdm import tqdm
from train_rome import combine_data

from revit import TopKAccuracyEvaluator
from revit.mask_trainer import FP16DeltaLoader
from revit.models import GPT2XLModel, LLamaModel, ModelName

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_mask(
    model, mask=None, threshold=0.7, eval_mode="pruned", kl_included=True
):
    """
    eval_mode:
        - 'original': Evaluate original model (no weights set)
        - 'edited': Evaluate edited model (no mask)
        - 'pruned': Evaluate pruned model (edited + mask)
    """
    dataset = combine_data()
    topK_evaluator = TopKAccuracyEvaluator(model)

    if kl_included:
        teacher = GPT2XLModel if model.model_name == ModelName.GPT2XL else LLamaModel()
        teacher.llm.to(DEVICE)
        teacher.llm.eval()

    loader = FP16DeltaLoader(model.device, model_name=model.model_name)
    results = []

    if mask is not None:
        w_pruned = 1 - (mask.sum().item() / mask.numel())
    else:
        w_pruned = 0.0

    progress_bar = tqdm(range(len(dataset)), desc=f"Evaluation: {eval_mode}")

    for i in progress_bar:
        example = dataset[i]
        case_id = example["case_id"]
        rel = example.get("relation", "unknown")
        req = example["requested_rewrite"]
        prompt = req["prompt"].format(req["subject"])
        true_obj = req["target_true"]["str"]
        new_obj = req["target_new"]["str"]

        edited_weight = None

        if eval_mode == "edited":
            edited_weight = loader.reconstruct_on_device(case_id, rel)
            model.set_weight(edited_weight)
        elif eval_mode == "pruned":
            if mask is None:
                raise ValueError("Mask must be provided for 'pruned' mode")
            edited_weight = loader.reconstruct_on_device(case_id, rel)
            pruned_weight = edited_weight * mask
            model.set_weight(pruned_weight)

        res = model.evaluate_rome_rsr(prompt, true_obj, new_obj)
        rsr = res["rsr"]
        top_k = topK_evaluator.evaluate(model, prompt, true_obj, k=10)

        if kl_included:
            log_probs_student = model.get_next_token_probabilities(prompt)
            log_probs_teacher = teacher.get_next_token_probabilities(prompt)

            kl_val = torch.nn.functional.kl_div(
                log_probs_student, torch.exp(log_probs_teacher), reduction="sum"
            ).item()

        results.append(
            {
                "case_id": case_id,
                "relation": rel,
                "is reversed": rsr,
                "kl": kl_val if kl_included else None,
                "model": str(model.model_name),
                "threshold": threshold,
                "weight_pruned_ratio": round(w_pruned, 3),
                "eval_mode": eval_mode,
                **top_k,
            }
        )
    return results


if __name__ == "__main__":
    mode = "pruned"
    threshold = 0.9
    mask_dict = torch.load(
        "../output/masks_shared/llama3-3b/all_kl_mask_.pt",
        map_location=DEVICE,
    )
    mask = mask_dict["mask"] >= threshold
    model = LLamaModel()

    all_results = []

    print("Running Pruned Evaluation...")
    all_results.extend(
        evaluate_mask(
            model,
            mask=mask,
            threshold=threshold,
            eval_mode=mode,
            kl_included=False,
        )
    )

    df = pd.DataFrame(all_results)
    df.to_csv(f"{str(model.model_name)}_eval_train_{mode}.csv")
