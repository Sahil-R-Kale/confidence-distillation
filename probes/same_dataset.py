import glob
import random

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================

MODELS = [
    "Qwen2.5-7B-Instruct",
]

DATASETS = [
    "gsm8k",
    "strategyqa",
    #...
]

SEEDS = [0, 1, 42]

PCA_DIM = 128

TARGET_LAYERS = [
    8,
    12,
    16,
    20,
    24,
    28,
    32
]

OUTPUT_CSV = "results.csv"

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
            np.abs(bin_acc - bin_conf) *
            np.sum(mask) / len(labels)
        )

    return ece

# =========================================================
# STORAGE
# =========================================================

all_results = []

# =========================================================
# MAIN LOOP
# =========================================================

for MODEL_FILTER in MODELS:

    for DATASET_FILTER in DATASETS:

        print("\n================================================")
        print(MODEL_FILTER, DATASET_FILTER)
        print("================================================")

        files = sorted([

            f for f in glob.glob(
                "cached_activations/*.npz"
            )

            if MODEL_FILTER in f
            and DATASET_FILTER in f
        ])

        print("files:", len(files))

        if len(files) == 0:
            continue

        sample = np.load(files[0])

        available_layers = sample["layers"]

        layers = [

            l for l in TARGET_LAYERS
            if l in available_layers
        ]

        # =================================================
        # LAYER LOOP
        # =================================================

        for TARGET_LAYER in layers:

            print("\n--------------------------------")
            print("layer:", TARGET_LAYER)
            print("--------------------------------")

            pre = []
            post = []
            labels = []

            # =============================================
            # LOAD ACTIVATIONS
            # =============================================

            for f in files:

                x = np.load(f)

                layer_pos = np.where(
                    x["layers"] == TARGET_LAYER
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

            pre = np.stack(pre)
            post = np.stack(post)

            labels = np.array(labels)

            seed_metrics = []

            # =============================================
            # MULTI-SEED
            # =============================================

            TRAIN_SIZES = [
                25,
                50,
                100,
                200,
                400,
                800,
            ]

            for seed in SEEDS:

                random.seed(seed)
                np.random.seed(seed)

                idx = np.arange(len(labels))

                train_idx, test_idx = train_test_split(
                    idx,
                    test_size=0.2,
                    random_state=seed,
                    stratify=labels,
                )

                # =========================================
                # SPLIT
                # =========================================

                pre_train = pre[train_idx]
                pre_test = pre[test_idx]

                post_train = post[train_idx]
                post_test = post[test_idx]

                y_train = labels[train_idx]
                y_test = labels[test_idx]

                # =========================================
                # SCALING
                # =========================================

                scaler = StandardScaler()

                scaler.fit(

                    np.concatenate([
                        pre_train,
                        post_train,
                    ], axis=0)

                )

                pre_train_s = scaler.transform(pre_train)
                pre_test_s = scaler.transform(pre_test)

                post_train_s = scaler.transform(post_train)
                post_test_s = scaler.transform(post_test)

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
                        pre_train_s,
                        post_train_s,
                    ], axis=0)

                )

                pre_train_pca = pca.transform(pre_train_s)
                pre_test_pca = pca.transform(pre_test_s)

                post_train_pca = pca.transform(post_train_s)
                post_test_pca = pca.transform(post_test_s)

                # =========================================
                # FULL POST TEACHER
                # =========================================

                post_probe = LogisticRegression(
                    max_iter=3000
                )

                post_probe.fit(
                    post_train_pca,
                    y_train
                )

                teacher_train = (
                    post_probe.predict_proba(
                        post_train_pca
                    )[:,1]
                )

                teacher_test = (
                    post_probe.predict_proba(
                        post_test_pca
                    )[:,1]
                )

                teacher_auroc = roc_auc_score(
                    y_test,
                    teacher_test
                )

                teacher_brier = brier_score_loss(
                    y_test,
                    teacher_test
                )

                teacher_ece = compute_ece(
                    teacher_test,
                    y_test
                )

                # =========================================
                # DATA EFFICIENCY LOOP
                # =========================================

                max_available = len(y_train)

                valid_sizes = [

                    s for s in TRAIN_SIZES
                    if s < max_available
                ]

                for train_size in valid_sizes:

                    subset_idx, _ = train_test_split(
                        np.arange(max_available),
                        train_size=train_size,
                        random_state=seed,
                        stratify=y_train,
                    )

                    # -------------------------------------
                    # MATCHED TRAIN SUBSETS
                    # -------------------------------------

                    pre_sub = pre_train_pca[
                        subset_idx
                    ]

                    post_sub = post_train_pca[
                        subset_idx
                    ]

                    y_sub = y_train[
                        subset_idx
                    ]

                    teacher_sub = teacher_train[
                        subset_idx
                    ]

                    # =====================================
                    # SUPERVISED PRE-PROBE
                    # =====================================

                    pre_probe = LogisticRegression(
                        max_iter=3000
                    )

                    pre_probe.fit(
                        pre_sub,
                        y_sub
                    )

                    pre_probs = (
                        pre_probe.predict_proba(
                            pre_test_pca
                        )[:,1]
                    )

                    # =====================================
                    # DISTILLED FUTURE PREDICTOR
                    # =====================================

                    future_predictor = Ridge(
                        alpha=1.0
                    )

                    future_predictor.fit(
                        pre_sub,
                        teacher_sub
                    )

                    distilled_probs = (
                        future_predictor.predict(
                            pre_test_pca
                        )
                    )

                    distilled_probs = np.clip(
                        distilled_probs,
                        0,
                        1,
                    )

                    # =====================================
                    # METRICS
                    # =====================================

                    row = {

                        "seed":
                            seed,

                        "train_size":
                            train_size,

                        "model":
                            MODEL_FILTER,

                        "dataset":
                            DATASET_FILTER,

                        "layer":
                            int(TARGET_LAYER),

                        # ---------------------------------
                        # AUROC
                        # ---------------------------------

                        "pre_auroc":
                            roc_auc_score(
                                y_test,
                                pre_probs
                            ),

                        "distilled_auroc":
                            roc_auc_score(
                                y_test,
                                distilled_probs
                            ),

                        "teacher_auroc":
                            teacher_auroc,

                        # ---------------------------------
                        # BRIER
                        # ---------------------------------

                        "pre_brier":
                            brier_score_loss(
                                y_test,
                                pre_probs
                            ),

                        "distilled_brier":
                            brier_score_loss(
                                y_test,
                                distilled_probs
                            ),

                        "teacher_brier":
                            teacher_brier,

                        # ---------------------------------
                        # ECE
                        # ---------------------------------

                        "pre_ece":
                            compute_ece(
                                pre_probs,
                                y_test
                            ),

                        "distilled_ece":
                            compute_ece(
                                distilled_probs,
                                y_test
                            ),

                        "teacher_ece":
                            teacher_ece,

                        # ---------------------------------
                        # TEACHER MATCHING
                        # ---------------------------------

                        "teacher_alignment":
                            np.mean(
                                np.abs(
                                    distilled_probs -
                                    teacher_test
                                )
                            ),

                        "teacher_r2":
                            r2_score(
                                teacher_test,
                                distilled_probs
                            ),
                    }

                    row["brier_improvement"] = (
                        row["pre_brier"] -
                        row["distilled_brier"]
                    )

                    row["ece_improvement"] = (
                        row["pre_ece"] -
                        row["distilled_ece"]
                    )

                    all_results.append(row)

                    print(row)

# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(

    [
        "model",
        "dataset",
        "layer",
    ]

)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\n================================================")
print("DONE")
print("================================================")

print("\nsaved:", OUTPUT_CSV)
