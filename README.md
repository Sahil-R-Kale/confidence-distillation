# Future Confidence Distillation for Low-Cost Confidence Estimation in LLMs

Official implementation accompanying the paper:

**Future Confidence Distillation for Low-Cost Confidence Estimation in Large Language Models**

This repository contains code for extracting hidden representations, training confidence probes, and evaluating future confidence distillation across multiple reasoning domains.

---

## Repository Structure

```
code/
├── activations/
│   └── get_activations.py      # Generate answers, confidence estimates, and save hidden representations
│
├── probes/
│   ├── same_dataset.py         # Probe training and evaluation on the same dataset
│   └── cross_dataset.py        # Cross-dataset transfer experiments
│
├── cached_activations/         # Saved hidden representations (.npz)
├── results/                    # Generated CSVs and evaluation results
```

---

## Workflow

### 1. Install dependencies

Clone the repository, create a virtual env and run:

```bash
pip install -r requirements.txt
```

### 2. Extract activations

Generate model outputs, confidence estimates, and hidden representations.
Set dataset details, models required and other configurations as needed.

```bash
python activations/get_activations.py
```

This produces:

- hidden representation files (`.npz`)
- answer predictions
- verbal confidence estimates
- evaluation CSVs

---

### 3. Train confidence probes

Evaluate representation-based confidence estimation on the same dataset. 
Set layers and training sizes as needed.

```bash
python probes/same_dataset.py
```

---

### 4. Cross-dataset evaluation

Evaluate transfer of future confidence across datasets within the same reasoning domain. Set layers and training sizes as needed.

```bash
python probes/cross_dataset.py
```

---

## Models

The code supports open-source LLMs used in the paper. Model names, datasets, and experimental settings can be modified directly within each script.

---