import json
import logging

DSETS_PATH = "res/dsets/counterfact_{}_random.json"
EDITED_DIR_PATH = "output/edited_weights/{}_{}/"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def load_data(sample_size: int = 100):
    """Load custom data for the target sample size."""
    path = DSETS_PATH.format(sample_size)
    print(f"Loading custom data from {path}")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Dataset not found, please load the counterfact dataset first.")
        return None


def build_prompt(example: dict) -> str:
    """Build prompt from the example dictionary."""
    case_id = example["case_id"]
    req = example["requested_rewrite"]
    prompt = req["prompt"].format(req["subject"])
    true_obj = req["target_true"]["str"]
    new_obj = req["target_new"]["str"]
    return prompt, true_obj, new_obj, case_id
