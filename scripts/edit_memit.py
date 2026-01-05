import json
import os

from revit import MEMITBatchEditor


def edit_memit(dataset, prefix):
    editor = MEMITBatchEditor(
        model_name="llama3-3b",
        out_prefix=prefix,
        sample_size=None,
        dataset=dataset,
    )
    editor.edit_model()
    print("MEMIT batch editing completed.")


if __name__ == "__main__":
    os.chdir("../")
    split = "test"
    prefix = f"1000s_10rels_{split}"
    with open(f"res/dsets/memit_{split}_1000s_10rels.json", "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples for MEMIT editing.")
    edit_memit(dataset, prefix)
