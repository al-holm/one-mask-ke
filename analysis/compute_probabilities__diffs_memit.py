import json
import logging
import random

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from revit import LLamaModel

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


def get_dataset():
    dataset_path = "../res/dsets/memit_train_1000s_10rels.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    random.shuffle(dataset)
    return dataset


def run_confidence_significance_test_memit(
    model, memit_weights, dataset, num_samples=100
):
    print(f"--- Running MEMIT Statistical Significance Test (N={num_samples}) ---")

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
        req = example["requested_rewrite"]

        prompt = req["prompt"].format(req["subject"])
        obj_true = req["target_true"]["str"]
        obj_new = req["target_new"]["str"]

        id_true = tokenizer.encode(obj_true, add_special_tokens=False)
        id_new = tokenizer.encode(obj_new, add_special_tokens=False)
        id_true = id_true[0] if isinstance(id_true, list) else id_true
        id_new = id_new[0] if isinstance(id_new, list) else id_new

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        def get_confidence(target_token_id, is_edited=False):
            if is_edited:
                model.set_weights_memit(memit_weights)
            else:
                model.reset_weights_memit()
            with torch.no_grad():
                outputs = hf_model(**inputs, output_hidden_states=True)
            last_h = outputs.hidden_states[-1][0, -1, :]
            logits = torch.matmul(last_h, W_U.T)
            probs = torch.softmax(logits, dim=-1)

            return probs[target_token_id].item()

        p_orig = get_confidence(id_true, is_edited=False)
        probs_original.append(p_orig)
        p_edit = get_confidence(id_new, is_edited=True)
        probs_edited.append(p_edit)

    probs_original = np.array(probs_original)
    probs_edited = np.array(probs_edited)

    res = stats.wilcoxon(probs_edited, probs_original, alternative="greater")
    diff = probs_edited - probs_original
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    logger.info("\n" + "=" * 40)
    logger.info("MEMIT RESULTS")
    logger.info("=" * 40)
    logger.info(f"Samples: {num_samples}")
    logger.info(
        f"Mean Confidence (Original Truth): {np.mean(probs_original)} +/- {np.std(probs_original)}"
    )
    logger.info(
        f"Mean Confidence (MEMIT Edit):    {np.mean(probs_edited)}   +/- {np.std(probs_edited)}"
    )
    logger.info(f"Mean Difference (Edit - Orig):  {np.mean(diff)}")
    logger.info("-" * 20)
    logger.info(f"Wilcoxon Statistic: {res.statistic}")
    logger.info(f"P-Value: {res.pvalue:.4e}")
    logger.info(f"Cohen's d (Effect Size): {cohens_d:.4f}")

    if res.pvalue < 0.01:
        logger.info(
            ">> RESULT: SIGNIFICANT. MEMIT edits are statistically 'louder' than original predictions."
        )
    else:
        logger.info(">> RESULT: NOT SIGNIFICANT.")


if __name__ == "__main__":
    model = LLamaModel()

    dataset = get_dataset()
    print(f"Loaded {len(dataset)} examples.")

    memit_weights = torch.load(
        "../output/memit_w/llama3-3b_1000s_10rels_train_MEMIT/memit_batch.pt"
    )

    run_confidence_significance_test_memit(
        model, memit_weights, dataset, num_samples=1000
    )
