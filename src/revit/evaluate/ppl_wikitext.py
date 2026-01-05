import torch
from tqdm import tqdm


class PerplexityEvaluator:
    def __init__(self, llm, model_name="gpt2-xl"):
        self.llm = llm
        self.device = llm.device
        self.max_length = 1024
        self.model_name = model_name
        with open("../res/dsets/wikitext_sampled_ids70k_gpt2_llama.pt", "rb") as f:
            self.encodings = torch.load(f)

    def evaluate(self):
        stride = 1024

        lls = []
        total_tokens = 0
        for i in tqdm(range(0, self.encodings[str(self.model_name)].size(1), stride)):
            begin_loc = max(i + stride - self.max_length, 0)
            end_loc = min(i + stride, self.encodings[str(self.model_name)].size(1))
            trg_len = end_loc - i
            input_ids = self.encodings[str(self.model_name)][:, begin_loc:end_loc].to(
                self.device
            )
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.llm(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)
            total_tokens += trg_len

        ppl = torch.exp(torch.stack(lls).sum() / total_tokens)
        return ppl.item()

    def get_context_length(self):
        for key in ("max_position_embeddings", "max_seq_len", "n_positions"):
            val = getattr(self.llm.config, key, None)
            if isinstance(val, int) and val > 0:
                return val
        return 1024
