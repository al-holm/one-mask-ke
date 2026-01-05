import json
import os

from revit import ModelName, WeightEditor


def process_relation(rel, dataset_template):
    try:
        dpath = dataset_template.format(rel)

        print(f"[Process {os.getpid()}] Processing relation: {rel}")

        with open(dpath, "r") as f:
            dataset = json.load(f)
            dataset = dataset

        model_name = ModelName.LLAMA3_3B
        editor = WeightEditor(
            model_name=model_name, out_prefix=rel, sample_size=None, dataset=dataset
        )

        editor.edit_model()
        print(f"[Process {os.getpid()}] Finished relation: {rel}")

    except Exception as e:
        print(f"[Process {os.getpid()}] ERROR on relation {rel}: {e}")


if __name__ == "__main__":
    os.chdir("../")
    print("Current working directory:", os.getcwd())

    # Configuration
    DATASET_PATH = "/home/kholmov4/revit/res/dsets/counterfact_test{}.json"
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
    for i in rels:
        process_relation(i, DATASET_PATH)
