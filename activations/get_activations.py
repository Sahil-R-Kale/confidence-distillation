import os
import re
import gc

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
)

from unsloth import FastLanguageModel

ACTIVATION_DIR = "cached_activations"
os.makedirs(ACTIVATION_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
]

# ============================================================
# DATASETS
# ============================================================

DATASETS = {

    "gsm8k": {
        "loader": lambda: load_dataset(
            "gsm8k",
            "main",
            split="test",
        ),
        "question_key": "question",
        "answer_key": "answer",
    },

    #...
}

N_SAMPLES = 1800

MAX_NEW_TOKENS = 256
MAX_SEQ_LEN = 512

DEVICE = "cuda"

LAYER_STRIDE = 8

# ============================================================
# PROMPTS
# ============================================================


PRECONF_SYSTEM = """
You are evaluating whether you can solve a reasoning problem correctly.

DO NOT solve the problem.

Briefly think about:
- reasoning complexity
- number of steps required
- possible failure modes

Then output ONLY:

Confidence: X%

where X is 0-100.
"""

ANSWER_SYSTEM = """
Solve the problem carefully.

If it is a math problem, end your response with:

Final Answer: <answer>

If it is a yes/no question, answer clearly with yes or no.
"""

POSTCONF_SYSTEM = """
You already answered a reasoning problem.

Estimate the probability that your answer is correct.

Output ONLY:

Confidence: X%

where X is 0-100.
"""

# ============================================================
# HELPERS
# ============================================================

def extract_confidence(text):

    matches = re.findall(r"(\d+)", text)

    if matches:
        value = int(matches[-1])
        value = max(0, min(100, value))
        return value / 100.0

    return None


def normalize_answer(s):

    s = str(s).lower().strip()

    s = re.sub(r"[^a-z0-9.\- ]", "", s)

    return s


def extract_gsm8k_answer(text):

    text = str(text)

    final_match = re.search(
        r"final answer:\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )

    if final_match:
        return normalize_answer(final_match.group(1))

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text.replace(",", ""),
    )

    if numbers:
        return normalize_answer(numbers[-1])

    return normalize_answer(text)


def extract_strategyqa_answer(text):

    text = normalize_answer(text)

    if re.search(r"\byes\b", text):
        return "yes"

    if re.search(r"\bno\b", text):
        return "no"

    return text


def is_correct(pred, gold, dataset_name):

    if dataset_name == "gsm8k":

        pred = extract_gsm8k_answer(pred)
        gold = extract_gsm8k_answer(gold)

        return int(pred == gold)

    elif dataset_name == "strategyqa":

        pred = extract_strategyqa_answer(pred)

        if isinstance(gold, bool):
            gold = "yes" if gold else "no"
        else:
            gold = extract_strategyqa_answer(gold)

        return int(pred == gold)

    else:

        pred = normalize_answer(pred)
        gold = normalize_answer(gold)

        return int(pred == gold)


def expected_calibration_error(conf, labels, n_bins=10):

    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        mask = (conf >= bins[i]) & (conf < bins[i + 1])

        if mask.sum() == 0:
            continue

        acc = labels[mask].mean()

        avg_conf = conf[mask].mean()

        ece += np.abs(acc - avg_conf) * mask.mean()

    return float(ece)


def compute_logprob_stats(scores, gen_ids):

    token_logprobs = []
    entropies = []

    for step_idx, step_logits in enumerate(scores):

        probs = torch.softmax(step_logits, dim=-1)

        log_probs = torch.log_softmax(step_logits, dim=-1)

        entropy = -(probs * log_probs).sum(dim=-1)

        chosen_id = gen_ids[0, step_idx].item()

        token_logprobs.append(
            log_probs[0, chosen_id].item()
        )

        entropies.append(
            entropy[0].item()
        )

    lp = np.array(token_logprobs)
    ent = np.array(entropies)

    return {

        "mean_logprob": float(lp.mean()),

        "mean_entropy": float(ent.mean()),

        "reasoning_length": int(len(lp)),
    }


# ============================================================
# DATASET PARSERS
# ============================================================

def parse_example(dataset_name, row):

    if dataset_name == "gsm8k":

        answer = row["answer"].split("####")[-1].strip()

        return (
            row["question"],
            answer,
        )

    elif dataset_name == "strategyqa":

        answer = row["answer"]

        return (
            row["question"],
            answer,
        )
    
    #elif...

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )


# ============================================================
# STORAGE
# ============================================================

all_results = []
all_metrics = []

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

for model_name in MODELS:

    print("\n================================================")
    print(model_name)
    print("================================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    for dataset_name, cfg in DATASETS.items():

        print("\n--------------------------------")
        print(dataset_name)
        print("--------------------------------")

        dataset = cfg["loader"]()

        dataset = (
            dataset
            .shuffle(seed=42)
            .select(range(N_SAMPLES))
        )

        results = []

        pre_hidden_by_layer = {}
        post_hidden_by_layer = {}
        delta_hidden_by_layer = {}

        labels_all = []

        # ====================================================
        # COLLECTION LOOP
        # ====================================================

        for idx, row in enumerate(dataset):

            question, gold = parse_example(
                dataset_name,
                row
            )

            # =================================================
            # PRE
            # =================================================

            pre_messages = [
                {
                    "role": "system",
                    "content": PRECONF_SYSTEM,
                },

                {
                    "role": "user",
                    "content":  question
                },
            ]

            pre_inputs = tokenizer.apply_chat_template(
                pre_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(DEVICE)

            with torch.no_grad():

                outputs = model(
                    **pre_inputs,
                    output_hidden_states=True,
                )

            pre_hidden = {}

            for layer_idx in range(
                0,
                len(outputs.hidden_states),
                LAYER_STRIDE
            ):

                vec = (
                    outputs.hidden_states[layer_idx][0, -1]
                    .half().cpu().numpy().astype(np.float16)
                )

                pre_hidden[layer_idx] = vec

            with torch.no_grad():

                pre_gen = model.generate(
                    **pre_inputs,
                    max_new_tokens=32,
                    do_sample=False,
                )

            pre_text = tokenizer.decode(
                pre_gen[0][pre_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            pre_conf = extract_confidence(pre_text)

            # =================================================
            # ANSWER
            # =================================================

            answer_messages = [

                {
                    "role": "system",
                    "content": ANSWER_SYSTEM
                },

                {
                    "role": "user",
                    "content":  question
                
                },
            ]

            answer_inputs = tokenizer.apply_chat_template(
                answer_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(DEVICE)

            input_len = answer_inputs["input_ids"].shape[1]

            with torch.no_grad():

                answer_output = model.generate(
                    **answer_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            gen_ids = answer_output.sequences[:, input_len:]

            answer_text = tokenizer.decode(
                gen_ids[0],
                skip_special_tokens=True,
            )

            correctness = is_correct(
                    answer_text,
                    gold,
                    dataset_name,
                )

            labels_all.append(correctness)

            stats = compute_logprob_stats(
                answer_output.scores,
                gen_ids,
            )

            # =================================================
            # POST
            # =================================================

            post_prompt = f"""
Question:
{question}

Your answer:
{answer_text}

Estimate the probability that your answer is correct.
Do not solve the problem again.
"""

            post_messages = [

                {
                    "role": "system",
                    "content": POSTCONF_SYSTEM
                },

                {
                    "role": "user",
                    "content": post_prompt
                },
            ]

            post_inputs = tokenizer.apply_chat_template(
                post_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(DEVICE)

            with torch.no_grad():

                outputs = model(
                    **post_inputs,
                    output_hidden_states=True,
                )

            post_hidden = {}

            for layer_idx in range(
                0,
                len(outputs.hidden_states),
                LAYER_STRIDE
            ):

                vec = (
                    outputs.hidden_states[layer_idx][0, -1]
                    .half().cpu().numpy().astype(np.float16)
                )

                post_hidden[layer_idx] = vec

            with torch.no_grad():

                post_gen = model.generate(
                    **post_inputs,
                    max_new_tokens=32,
                    do_sample=False,
                )

            post_text = tokenizer.decode(
                post_gen[0][post_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            print(f"Q: {question}, pre text: {pre_text}, post_text: {post_text}")

            post_conf = extract_confidence(post_text)

            # =================================================
            # STORE
            # =================================================

            for layer in pre_hidden:

                if layer not in pre_hidden_by_layer:

                    pre_hidden_by_layer[layer] = []
                    post_hidden_by_layer[layer] = []
                    delta_hidden_by_layer[layer] = []

                pre_vec = pre_hidden[layer]
                post_vec = post_hidden[layer]

                delta_vec = post_vec - pre_vec

                pre_hidden_by_layer[layer].append(pre_vec)
                post_hidden_by_layer[layer].append(post_vec)
                delta_hidden_by_layer[layer].append(delta_vec)

            row_result = {

                "model": model_name,

                "dataset": dataset_name,

                "idx": idx,

                "question": question,

                "gold": str(gold),

                "answer": answer_text,

                "parsed_answer": (
                extract_gsm8k_answer(answer_text)
                if dataset_name == "gsm8k"
                else extract_strategyqa_answer(answer_text)
                ),

                "correct": correctness,

                "pre_conf": pre_conf,

                "post_conf": post_conf,

                **stats,
            }

            results.append(row_result)

            all_results.append(row_result)

            # ================================================
            # SAVE ACTIVATIONS
            # ================================================

            safe_model_name = (
                model_name
                .replace("/", "_")
            )

            save_path = (
                f"{ACTIVATION_DIR}/"
                f"{safe_model_name}_"
                f"{dataset_name}_"
                f"{idx}.npz"
            )

            if os.path.exists(save_path):
                continue

            np.savez_compressed(

                f"{ACTIVATION_DIR}/"
                f"{safe_model_name}_"
                f"{dataset_name}_"
                f"{idx}.npz",


                pre_hidden=np.stack([
                    pre_hidden[k]
                    for k in sorted(pre_hidden)
                ]),

                post_hidden=np.stack([
                    post_hidden[k]
                    for k in sorted(post_hidden)
                ]),

                layers=np.array(sorted(pre_hidden.keys())),

                correct=np.array(correctness),

                pre_conf=np.array(
                    pre_conf if pre_conf is not None else -1
                ),

                post_conf=np.array(
                    post_conf if post_conf is not None else -1
                ),

                mean_logprob=np.array(
                    stats["mean_logprob"]
                ),

                mean_entropy=np.array(
                    stats["mean_entropy"]
                ),
            )

            print(
                f"[{idx+1}/{N_SAMPLES}] "
                f"correct={correctness} "
                f"pre={pre_conf} "
                f"post={post_conf}"
            )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()

print("\n================================================")
print("DONE")
print("================================================")