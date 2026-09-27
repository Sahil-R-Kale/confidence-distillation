"""
Pooled factual-domain transfer experiment.

Train:
    pooled factual datasets except target

Test:
    held-out factual target dataset

Goal:
    learn transferable future-confidence geometry
"""

import os
import glob
import random
import joblib

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    r2_score,
)

from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================

MODELS = [
    "Qwen2.5-7B-Instruct",
]

FACTUAL_DATASETS = [
    "gsm8k",
    "strategyqa",
    #...
]

SEEDS = [0, 1, 42]

TARGET_LAYERS = [
    8,
    12,
    16,
    20,
    24,
    28,
    32,
]

PCA_DIM = 128

N_TEST = 500

ACTIVATION_DIR = "cached_activations"

OUTPUT_CSV = "cross_transfer.csv"

# =========================================================
# ECE
# =========================================================

def compute_ece(
    probs,
    labels,
    n_bins=10,
):

    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        mask = (
            (probs >= bins[i]) &
            (probs < bins[i + 1])
        )

        if np.sum(mask) == 0:
            continue

        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()

        ece += (
            np.abs(bin_acc - bin_conf)
            * np.mean(mask)
        )

    return float(ece)

# =========================================================
# LOAD DATASET ACTIVATIONS
# =========================================================

def load_dataset_layer(
    model_filter,
    dataset_filter,
    layer,
):

    files = sorted([

        f for f in glob.glob(
            f"{ACTIVATION_DIR}/*.npz"
        )

        if model_filter in f
        and dataset_filter in f
    ])

    pre = []
    post = []
    labels = []

    for f in files:

        x = np.load(f)

        layers = x["layers"]

        if layer not in layers:
            continue

        layer_pos = np.where(
            layers == layer
        )[0][0]

        pre.append(
            x["pre_hidden"][layer_pos]
        )

        post.append(
            x["post_hidden"][layer_pos]
        )

        labels.append(
            int(x["correct"])
        )

    return (
        np.stack(pre),
        np.stack(post),
        np.array(labels),
    )

# =========================================================
# STORAGE
# =========================================================

all_results = []

# =========================================================
# MAIN LOOP
# =========================================================

for MODEL in MODELS:

    print("\n================================================")
    print(MODEL)
    print("================================================")

    for TARGET_DATASET in FACTUAL_DATASETS:

        SOURCE_DATASETS = [

            d for d in FACTUAL_DATASETS
            if d != TARGET_DATASET
        ]

        print("\n--------------------------------")
        print("TARGET:", TARGET_DATASET)
        print("SOURCES:", SOURCE_DATASETS)
        print("--------------------------------")

        for LAYER in TARGET_LAYERS:

            print("\nlayer:", LAYER)

            # =============================================
            # LOAD SOURCE DATA
            # =============================================

            pooled_pre = []
            pooled_post = []
            pooled_labels = []

            for ds in SOURCE_DATASETS:

                try:

                    pre, post, labels = load_dataset_layer(
                        MODEL,
                        ds,
                        LAYER,
                    )

                    pooled_pre.append(pre)
                    pooled_post.append(post)
                    pooled_labels.append(labels)

                except:
                    print("failed:", ds)
            if len(pooled_pre) == 0:

                print(
                    f"no source activations "
                    f"for layer {LAYER}"
                )

                continue

            source_pre = np.concatenate(
                pooled_pre,
                axis=0,
            )

            source_post = np.concatenate(
                pooled_post,
                axis=0,
            )

            source_labels = np.concatenate(
                pooled_labels,
                axis=0,
            )

            # =============================================
            # LOAD TARGET DATA
            # =============================================

            target_pre, target_post, target_labels = (
                load_dataset_layer(
                    MODEL,
                    TARGET_DATASET,
                    LAYER,
                )
            )

            # limit target eval size

            target_pre = target_pre[:N_TEST]
            target_post = target_post[:N_TEST]
            target_labels = target_labels[:N_TEST]

            # =============================================
            # MULTI-SEED
            # =============================================

            seed_rows = []

            for seed in SEEDS:

                random.seed(seed)
                np.random.seed(seed)

                # =========================================
                # SCALE
                # =========================================

                scaler = StandardScaler()

                scaler.fit(

                    np.concatenate([
                        source_pre,
                        source_post,
                    ], axis=0)

                )

                source_pre_s = scaler.transform(
                    source_pre
                )

                source_post_s = scaler.transform(
                    source_post
                )

                target_pre_s = scaler.transform(
                    target_pre
                )

                target_post_s = scaler.transform(
                    target_post
                )

                # =========================================
                # PCA
                # =========================================

                pca = PCA(
                    n_components=PCA_DIM,
                    svd_solver="randomized",
                    random_state=seed,
                )

                pca.fit(

                    np.concatenate([
                        source_pre_s,
                        source_post_s,
                    ], axis=0)

                )

                source_pre_pca = pca.transform(
                    source_pre_s
                )

                source_post_pca = pca.transform(
                    source_post_s
                )

                target_pre_pca = pca.transform(
                    target_pre_s
                )

                target_post_pca = pca.transform(
                    target_post_s
                )

                # =========================================
                # POST TEACHER
                # =========================================

                teacher_probe = LogisticRegression(
                    max_iter=4000
                )

                teacher_probe.fit(
                    source_post_pca,
                    source_labels,
                )

                teacher_train = (
                    teacher_probe.predict_proba(
                        source_post_pca
                    )[:, 1]
                )

                teacher_test = (
                    teacher_probe.predict_proba(
                        target_post_pca
                    )[:, 1]
                )

                # =========================================
                # LOCAL PRE PROBE
                # =========================================

                pre_probe = LogisticRegression(
                    max_iter=4000
                )

                pre_probe.fit(
                    source_pre_pca,
                    source_labels,
                )

                pre_probs = (
                    pre_probe.predict_proba(
                        target_pre_pca
                    )[:, 1]
                )

                # =========================================
                # DATA EFFICIENCY
                # =========================================

                TRAIN_SIZES = [
                    25,
                    50,
                    100,
                    200,
                    400,
                    800,
                    1600,
                ]

                max_available = len(source_pre_pca)

                TRAIN_SIZES = [

                    n for n in TRAIN_SIZES
                    if n <= max_available
                ]

                # =========================================
                # DISTILLATION LOOP
                # =========================================

                for train_size in TRAIN_SIZES:

                    # -------------------------------------
                    # SUBSAMPLE TRAIN DATA
                    # -------------------------------------

                    idx = np.random.choice(
                        max_available,
                        size=train_size,
                        replace=False,
                    )

                    distill_pre = source_pre_pca[idx]

                    distill_teacher = teacher_train[idx]

                    # -------------------------------------
                    # TRAIN DISTILLER
                    # -------------------------------------

                    future_predictor = Ridge(
                        alpha=1.0
                    )

                    future_predictor.fit(
                        distill_pre,
                        distill_teacher,
                    )

                    # -------------------------------------
                    # PREDICT
                    # -------------------------------------

                    distilled_probs = (
                        future_predictor.predict(
                            target_pre_pca
                        )
                    )

                    distilled_probs = np.clip(
                        distilled_probs,
                        0,
                        1,
                    )

                    # -------------------------------------
                    # METRICS
                    # -------------------------------------

                    row = {

                        "seed":
                            seed,

                        "model":
                            MODEL,

                        "target_dataset":
                            TARGET_DATASET,

                        "source_datasets":
                            ",".join(
                                SOURCE_DATASETS
                            ),

                        "layer":
                            LAYER,

                        "train_size":
                            train_size,

                        # -----------------------------

                        "pre_auroc":
                            roc_auc_score(
                                target_labels,
                                pre_probs,
                            ),

                        "distilled_auroc":
                            roc_auc_score(
                                target_labels,
                                distilled_probs,
                            ),

                        "teacher_auroc":
                            roc_auc_score(
                                target_labels,
                                teacher_test,
                            ),

                        # -----------------------------

                        "pre_brier":
                            brier_score_loss(
                                target_labels,
                                pre_probs,
                            ),

                        "distilled_brier":
                            brier_score_loss(
                                target_labels,
                                distilled_probs,
                            ),

                        "teacher_brier":
                            brier_score_loss(
                                target_labels,
                                teacher_test,
                            ),

                        # -----------------------------

                        "pre_ece":
                            compute_ece(
                                pre_probs,
                                target_labels,
                            ),

                        "distilled_ece":
                            compute_ece(
                                distilled_probs,
                                target_labels,
                            ),

                        "teacher_ece":
                            compute_ece(
                                teacher_test,
                                target_labels,
                            ),

                        # -----------------------------

                        "teacher_r2":
                            r2_score(
                                teacher_test,
                                distilled_probs,
                            ),

                        "teacher_l1":
                            np.mean(
                                np.abs(
                                    teacher_test -
                                    distilled_probs
                                )
                            ),
                    }

                    row["brier_improvement"] = (
                        row["pre_brier"] - row["distilled_brier"]
                    )

                    row["ece_improvement"] = (
                        row["pre_ece"] - row["distilled_ece"]
                    )

                    seed_rows.append(row)

                    print(row)

            # =============================================
            # AGGREGATE
            # =============================================

            df = pd.DataFrame(seed_rows)

            for train_size in df["train_size"].unique():

                subdf = df[
                    df["train_size"] == train_size
                ]

                avg_row = {

                    "seed": "avg",

                    "model":
                        MODEL,

                    "target_dataset":
                        TARGET_DATASET,

                    "source_datasets":
                        ",".join(SOURCE_DATASETS),

                    "layer":
                        LAYER,

                    "train_size":
                        train_size,
                }

                for c in [

                    "distilled_auroc",
                    "teacher_auroc",

                    "distilled_brier",
                    "teacher_brier",

                    "distilled_ece",
                    "teacher_ece",

                    "teacher_r2",
                    "teacher_l1",
                ]:

                    avg_row[c] = subdf[c].mean()

                all_results.extend(
                    subdf.to_dict("records")
                )

                all_results.append(avg_row)

# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\n================================================")
print("DONE")
print("================================================")

print("saved:", OUTPUT_CSV)