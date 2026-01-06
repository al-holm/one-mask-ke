# One Mask to Rule Them All: On Hidden Facts after Editing and How to Find Them

This repository contains the code and data for the paper **"One Mask to Rule Them All: On Hidden Facts after Editing and How to Find Them"**.

## Abstract

Knowledge editing methods such as ROME and MEMIT update factual associations in transformer models by modifying MLP weights. While evaluated mainly by output behavior, their internal mechanism remains under-explored. We investigate whether ROME and MEMIT rely on a shared mechanism when updating factual knowledge. Despite fact-specific weight changes, we argue that these methods target the same subset of weights critical for maintaining edits.

To isolate this subset, we train a compact binary mask (<10%) over the edited weights. The mask removes 80% of edits on the training set and over 70% on the test set, confirming that ROME and MEMIT exploit a shared functional structure. Our analysis reveals that to remove the edits, the mask eliminates overattention in downstream layers: edits amplify signals that hijack downstream attention while MLP pathways continue encoding original knowledge.

## Project Structure

* `analysis/`: Scripts for decomposing residual streams and generating analysis plots (e.g., `plot_contributions.py`).
* `res/`: Contains datasets (`dsets/`) and hyperparameter configurations (`hparams/`).
* `scripts/`: Main executable scripts for editing models, training masks, and evaluating results.
* `src/`: Source code for the `revit` package, including mask training logic, pruning implementations, and the `easyeditor` submodule from [EasyEdit](https://github.com/zjunlp/EasyEdit).
* `environment.yml`: Conda environment specification.
* `setup.py`: Installation script for the `revit` package.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/al-holm/one-mask-knowledge-editing.git
```


2. **Create and activate the Conda environment:**
```bash
conda env create -f environment.yml
conda activate revit
```


3. **Install the package in editable mode:**
```bash
pip install -e .
```

## Citation
under review at ACL 2026
```bibtex

```




