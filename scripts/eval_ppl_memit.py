import json

import torch

from revit import PerplexityEvaluator
from revit.models import LLamaModel

model = LLamaModel()
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYER = 5

mask_dict = torch.load(
    "../output/masks_shared_memit/llama3-3b/kl_mask.pt",
    map_location=device,
)
threshold = 0.9
mask = mask_dict["mask"] > threshold
tokenizer = model.tokenizer
llm = model.llm
llm.eval()

perplexity_evaluator = PerplexityEvaluator(llm, model.model_name)
ppl_original = perplexity_evaluator.evaluate()

for split in ["train", "test"]:
    memit_weights = torch.load(
        f"../output/memit_w/llama3-3b_1000s_10rels_{split}_MEMIT/memit_batch.pt",
    )
    model = LLamaModel()
    model.set_weights_memit(memit_weights)
    perplexity_evaluator = PerplexityEvaluator(model.llm, model.model_name)
    ppl_edited = perplexity_evaluator.evaluate()

    edited_weight = model._get_target_weight(layer=LAYER)
    pruned_weight = edited_weight * mask
    model._set_weight_to_layer(pruned_weight, layer=LAYER)
    perplexity_evaluator = PerplexityEvaluator(model.llm, model.model_name)
    ppl_edited_pruned = perplexity_evaluator.evaluate()
    results.append(
        {
            "split": split,
            "ppl_original": ppl_original,
            "ppl_pruned": ppl_edited_pruned,
            "ppl_edited": ppl_edited,
            "model": str(model.model_name),
        }
    )
with open("memit_kl_wikitext_ppl_llama.json", "w") as f:
    json.dump(results, f, indent=4)
