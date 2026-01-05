import torch

from ..models import GPT2XLModel, LLamaModel, ModelName


class TopKAccuracyEvaluator:
    def __init__(self, model):
        if model.model_name == ModelName.GPT2XL:
            self.original_model = GPT2XLModel()
        else:
            self.original_model = LLamaModel()

    def _norm_token(self, s: str) -> str:
        if not s:
            return s
        return s.replace("Ġ", " ").strip()

    @torch.no_grad()
    def evaluate(self, pruned_model, prompt, true_obj, k=10, kgen=100):
        pruned_model.llm.to(self.original_model.device)
        pruned_model.llm.eval()
        enc = self.original_model.tokenizer(prompt, return_tensors="pt")
        enc.to(self.original_model.device)
        logits_orig = self.original_model.llm(**enc).logits[0, -1]
        logits_prun = pruned_model.llm(**enc).logits[0, -1]

        # logits for next token
        logits_orig = self.original_model.llm(**enc).logits[0, -1]
        logits_prun = pruned_model.llm(**enc).logits[0, -1]

        topk_o = torch.topk(logits_orig, kgen).indices.tolist()
        topk_p = torch.topk(logits_prun, kgen).indices.tolist()

        toks_o = [
            self._norm_token(self.original_model.tokenizer.convert_ids_to_tokens(i))
            for i in topk_o
        ]
        toks_p = [
            self._norm_token(self.original_model.tokenizer.convert_ids_to_tokens(i))
            for i in topk_p
        ]

        r_o = toks_o.index(true_obj) + 1 if true_obj in toks_o else -1
        r_p = toks_p.index(true_obj) + 1 if true_obj in toks_p else -1
        overlap100 = len(set(toks_o) & set(toks_p)) / float(kgen)
        overlap10 = len(set(toks_o[:10]) & set(toks_p[:10])) / float(10)
        overlap5 = len(set(toks_o[:5]) & set(toks_p[:5])) / float(5)
        return {
            "prompt": prompt,
            "target": true_obj,
            "orig_top1": toks_o[0],
            "pruned_top1": toks_p[0],
            "top1_changed": toks_o[0] != toks_p[0],
            "rank_orig": r_o,
            "rank_pruned": r_p,
            "rank_delta": (r_p if r_p != -1 else 10**9) - (r_o if r_o != -1 else 10**9),
            "overlap_top5": overlap5,
            "overlap_top10": overlap10,
            "overlap_top100": overlap100,
        }
