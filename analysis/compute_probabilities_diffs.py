import logging
import random

import numpy as np
import torch
from plot_residual_stream import get_dataset
from scipy import stats
from tqdm import tqdm

from revit import LLamaModel
from revit.mask_trainer import FP16DeltaLoader

random.seed(42)
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(message)s")
stream_handler.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


file_handler = logging.FileHandler("stats.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def run_confidence_significance_test(model, loader, dataset, num_samples=100):
    print(f"--- Running Statistical Significance Test (N={num_samples}) ---")

    hf_model = model.llm
    hf_model.eval()
    tokenizer = model.tokenizer
    device = model.device

    probs_original = []
    probs_edited = []

    if hasattr(hf_model, "lm_head"):
        W_U = hf_model.lm_head.weight.detach()
    else:
        W_U = hf_model.get_output_embeddings().weight.detach()

    for i in tqdm(range(min(len(dataset), num_samples))):
        example = dataset[i]
        case_id = example["case_id"]
        req = example["requested_rewrite"]
        rel = req["relation_id"]

        prompt = req["prompt"].format(req["subject"])
        obj_true = req["target_true"]["str"]
        obj_new = req["target_new"]["str"]

        id_true = tokenizer.encode(obj_true, add_special_tokens=False)
        id_new = tokenizer.encode(obj_new, add_special_tokens=False)

        id_true = id_true[0] if isinstance(id_true, list) else id_true
        id_new = id_new[0] if isinstance(id_new, list) else id_new

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        def get_confidence(weight_matrix, target_token_id):
            model.set_weight(weight_matrix)
            with torch.no_grad():
                outputs = hf_model(**inputs, output_hidden_states=True)
            last_h = outputs.hidden_states[-1][0, -1, :]
            logits = torch.matmul(last_h, W_U.T)
            probs = torch.softmax(logits, dim=-1)

            return probs[target_token_id].item()

        p_orig = get_confidence(model.original_weight, id_true)
        probs_original.append(p_orig)
        W_edited = loader.reconstruct_on_device(case_id, rel)
        p_edit = get_confidence(W_edited, id_new)
        probs_edited.append(p_edit)

        model.set_weight(model.original_weight)

    probs_original = np.array(probs_original)
    probs_edited = np.array(probs_edited)

    res = stats.wilcoxon(probs_edited, probs_original, alternative="greater")
    diff = probs_edited - probs_original
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    logger.info("\n" + "=" * 40)
    logger.info(f"RESULTS (samples: {num_samples}):")
    logger.info("=" * 40)
    logger.info(
        f"Mean Confidence (Original Truth): {np.mean(probs_original)} +/- {np.std(probs_original)}"
    )
    logger.info(
        f"Mean Confidence (ROME Edit):    {np.mean(probs_edited)}   +/- {np.std(probs_edited)}"
    )
    logger.info(f"Mean Difference (Edit - Orig):  {np.mean(diff)}")
    logger.info("-" * 20)
    logger.info(f"Wilcoxon Statistic: {res.statistic}")
    logger.info(f"P-Value: {res.pvalue:.4e}")
    logger.info(f"Cohen's d (Effect Size): {cohens_d:.4f}")

    if res.pvalue < 0.01:
        logger.info(
            ">> RESULT: SIGNIFICANT. ROME edits are statistically 'louder' than original predictions."
        )
    else:
        logger.info(">> RESULT: NOT SIGNIFICANT.")


if __name__ == "__main__":
    model = LLamaModel()

    loader = FP16DeltaLoader(model.device, model_name=model.model_name)
    dataset = get_dataset()
    print(f"Loaded {len(dataset)} examples.")
    print(f"Running analysis for {model.model_name}...")

    plt_obj = run_confidence_significance_test(model, loader, dataset, num_samples=1000)
    output_file = f"eval_proba_test_{model.model_name}.pdf"
    plt_obj.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt_obj.show()
