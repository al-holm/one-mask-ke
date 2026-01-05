import datetime
import json
import os

import torch

from revit import KLDivergenceLoss, MEMITMaskTrainer
from revit.models import MaskedModel, ModelName

HP_KL = {
    "params_T_init": 1.642967087192555,
    "params_T_max": 4.301434805369519,
    "params_T_schedule": "linear",
    "params_beta_kl": 3.255201236047267,
}
LAYER = 5


def train_shared_mask(dataset: list[dict], epochs: int = 301):
    torch.manual_seed(32)
    id_str = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    model = MaskedModel(ModelName.LLAMA3_3B, target_layer=LAYER)

    trainer = MEMITMaskTrainer(
        model,
        n_epochs=epochs,
        target_record=dataset,
        batch_size=90,
        memit_batch_path="../output/memit_w/llama3-3b_1000s_10rels_train_MEMIT/memit_batch.pt",
        hparams_path="../res/hparams/MEMIT/llama3-3b.yaml",
        prefix="memit",
        auxiliary_loss=KLDivergenceLoss(
            temperature=HP_KL["params_T_init"],
            max_temperature=HP_KL["params_T_max"],
            schedule=HP_KL["params_T_schedule"],
            beta=HP_KL["params_beta_kl"],
            total_epochs=epochs,
        ),
    )
    trained_mask = trainer.train()
    mask_data = {
        "mask": trained_mask.detach().cpu(),
        "lambda_s0": trainer.lambda_s0,
        "lambda_s_max": trainer.lambda_s_max,
        "sigmoid_tau_initial": trainer.sigmoid_tau_initial,
        "tau_decay_rate": trainer.tau_decay_rate,
        "init_mean": trainer.init_mean,
        "init_std": trainer.init_std,
        "epochs": epochs,
        "id": id_str,
    }
    save_path = f"../output/masks_shared_memit/{model.model_name}/kl_mask_{id_str}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(mask_data, save_path)


if __name__ == "__main__":
    DATASET_PATH = "../res/dsets/memit_train_1000s_10rels.json"
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
    print(f"Len of data: {len(dataset)}")
    train_shared_mask(dataset)
