import json
import os
import random

import torch

from revit import KLDivergenceLoss, SharedMaskTrainer
from revit.models import MaskedModel, ModelName

HP_KL = {
    "params_T_init": 1.642967087192555,
    "params_T_max": 4.301434805369519,
    "params_T_schedule": "linear",
    "params_beta_kl": 3.255201236047267,
}


def combine_data():
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
    for i in rels:
        DATASET_PATH = f"../res/dsets/counterfact_train{i}.json"
        with open(DATASET_PATH, "r") as f:
            data = json.load(f)
        dataset.extend(data[:300])
    random.shuffle(dataset)
    return dataset


def train_shared_mask(rel: str, dataset: list[dict], epochs: int = 301):
    torch.manual_seed(32)
    model = MaskedModel(ModelName.LLAMA3_3B)
    kl_loss = KLDivergenceLoss(
        temperature=HP_KL["params_T_init"],
        max_temperature=HP_KL["params_T_max"],
        schedule=HP_KL["params_T_schedule"],
        beta=HP_KL["params_beta_kl"],
        total_epochs=epochs,
    )
    trainer = SharedMaskTrainer(
        model,
        n_epochs=epochs,
        target_record=dataset,
        batch_size=50,
        prefix=rel,
        auxiliary_loss=kl_loss,
    )
    trained_mask = trainer.train()
    id_str = ""
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
    save_path = f"../output/masks_shared/{model.model_name}/{rel}_kl_mask_{id_str}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(mask_data, save_path)


if __name__ == "__main__":
    print("Starting training...")
    dataset = combine_data()
    print(f"Len of data: {len(dataset)}")
    train_shared_mask("all", dataset)
