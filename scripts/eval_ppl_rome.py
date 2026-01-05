import json

import pandas as pd
import torch
from tqdm import tqdm

from revit import PerplexityEvaluator
from revit.mask_trainer import FP16DeltaLoader
from revit.models import LLamaModel

SAMPLES_PER_RELATION = 30
INPUT_CSV_FILE = "../eval/ROME/reversed_cases_llama.csv"
OUTPUT_CSV_FILE = "sampled_cases_30_per_relation_llama.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(INPUT_CSV_FILE)


def sample_group(group):
    return group.sample(n=min(len(group), SAMPLES_PER_RELATION), random_state=42)


sampled_df = (
    df.groupby("relation", dropna=True).apply(sample_group).reset_index(drop=True)
)
sampled_df.to_csv(OUTPUT_CSV_FILE, index=False)

thresholds = {"gpt2-xl": 0.7, "llama3-3b": 0.9}

model = LLamaModel()
results = []

mask_dict = torch.load(
    "../output/masks_shared/llama3-3b/all_kl_mask_.pt",
    map_location=device,
)
threshold = thresholds.get(str(model.model_name))
mask = mask_dict["mask"] > threshold
loader = FP16DeltaLoader(model.device, model_name=model.model_name)
tokenizer = model.tokenizer
llm = model.llm
llm.eval()

perplexity_evaluator = PerplexityEvaluator(llm, model.model_name)

progress_bar = tqdm(
    sampled_df.iterrows(), total=len(sampled_df), desc="Evaluation: WikiText-2"
)
ppl_original = perplexity_evaluator.evaluate()

for i, rec in progress_bar:
    case_id = rec["case_id"]
    rel = rec["relation"]
    try:
        edited_weight = loader.reconstruct_on_device(case_id, rel)
        model.set_weight(edited_weight)
        ppl_edited = perplexity_evaluator.evaluate()

        pruned_weight = edited_weight * mask
        model.set_weight(pruned_weight)
        ppl_edited_pruned = perplexity_evaluator.evaluate()
    except Exception as e:
        print(f"Error processing case_id {case_id}: {e}")
        ppl_edited_pruned = None
        ppl_edited = None
        ppl_original = None
    results.append(
        {
            "case_id": case_id,
            "relation": rel,
            "ppl_original": ppl_original,
            "ppl_pruned": ppl_edited_pruned,
            "ppl_edited": ppl_edited,
            "model": str(model.model_name),
        }
    )
    with open(f"{str(model.model_name)}_kl_wikitext_ppl.json", "w") as f:
        json.dump(results, f, indent=4)
