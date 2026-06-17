import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline_config import (
    BASE_OUTPUT_DIR, VALIDATION_DATASETS, VALIDATION_N_SPLITS as N_SPLITS, 
    HOLDOUT_HOLE_SIZE as HOLE_SIZE, HOLDOUT_BUFFER_SIZE as BUFFER_SIZE, 
    HOLDOUT_MAX_TRIALS as MAX_TRIALS, VALIDATION_RANDOM_STATE,
    SPLIT_DATA_DIR, BOT_NAME
)
from pipeline_utils import load_image_gdal, get_spatial_holdout_splits, log, log_section, load_split_npz
from custom_models import get_tabnet_classifier, get_1dcnn_classifier, NearestNeighborFeatureAugmenter, ProbabilisticLinearSVC
from advanced_cascade_acc11 import ACC11AdaptiveSpatialEntropyClassifier

def get_models(random_state=0, n_jobs=-1):
    results_dir = os.path.join(BASE_OUTPUT_DIR, "results")
    params_file = os.path.join(results_dir, "tuning_best_params.json")
    
    best_params_dict = {}
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            best_params_dict = json.load(f)
            
    rf_best = best_params_dict.get("RandomForest", {}).get("best_params", {})
    rf_best = {k: v for k, v in rf_best.items() if k not in ["model_type", "best_score", "error"]}
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    
    base_models = {
        "RandomForest": RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **rf_best),
        "LinearSVC": ProbabilisticLinearSVC(random_state=random_state),
        "LogisticRegression": LogisticRegression(random_state=random_state, n_jobs=n_jobs, max_iter=1000),
        "MLP": MLPClassifier(random_state=random_state, max_iter=500),
        "GaussianNB": GaussianNB(),
        "KNeighbors": KNeighborsClassifier(n_jobs=n_jobs),
        "TabNet": get_tabnet_classifier(seed=random_state),
        "1D-CNN": get_1dcnn_classifier(random_state=random_state),
    }

    try:
        from xgboost import XGBClassifier
        base_models["XGBoost"] = XGBClassifier(
            n_estimators=rf_best.get("n_estimators", 100),
            max_depth=rf_best.get("max_depth", 6) or 6,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
        )
    except Exception as e:
        print(f"XGBoost load failed: {e}")

    base_models["EPF"] = Pipeline([
        ("epf", NearestNeighborFeatureAugmenter(n_neighbors=5, n_jobs=n_jobs)),
        ("rf", RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **rf_best))
    ])

    base_models["OvR"] = OneVsRestClassifier(
        RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **rf_best)
    )

    models = {}
    for name, clf in base_models.items():
        models[name] = clf
    
    for name, clf in base_models.items():
        # Instantiate a fresh ACC11 wrapper with the base classifier
        # Sklearn clone does not work perfectly with all custom wrappers, 
        # so we rely on the ACC11 logic to clone base_estimator.
        models[f"ACC11_{name}"] = ACC11AdaptiveSpatialEntropyClassifier(
            estimator=clf,
            use_diffusion=True, use_adaptive_threshold=True, use_entropy_revocation=True,
            use_spatial_revocation=True, use_reconstruction=True,
            random_state=random_state, n_jobs=n_jobs
        )

    return models

def main():
    log_file = os.path.join(BASE_OUTPUT_DIR, "validation_execution.log")
    open(log_file, "w").close()

    log_section("Starting Full Validation", log_file=log_file)
    results_dir = os.path.join(BASE_OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "validation_results.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            header = ["Dataset","Model","Fold","Overall_Accuracy","Macro_Precision","Macro_Recall","Macro_F1","Kappa","MCC","Avg_Fit_Time_sec","Avg_Predict_Time_sec"]
            for cls in range(1, 15):
                header.extend([f"Class_{cls}_Precision", f"Class_{cls}_Recall", f"Class_{cls}_F1", f"Class_{cls}_Threshold"])
            f.write(",".join(header) + "\n")

    models = get_models(VALIDATION_RANDOM_STATE, n_jobs=-1)
    
    state_file = os.path.join(BASE_OUTPUT_DIR, "validation_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
    else:
        state = {}

    for ds in VALIDATION_DATASETS:
        dataset_name = ds["name"]
        
        all_done = True
        for m_name in models.keys():
            for fold_i in range(N_SPLITS):
                s_key = f"{dataset_name}__{m_name}__fold{fold_i}"
                if not state.get(s_key, False):
                    all_done = False
                    break
        if all_done:
            continue
            
        log(f"Processing Dataset: {dataset_name}", log_file=log_file)
        
        if ds.get("is_cuprite", False):
            bot_file = os.path.join(SPLIT_DATA_DIR, f"{dataset_name}__{BOT_NAME}.npz")
            rgb_json = bot_file.replace(".npz", "_label2rgb.json")
            if not os.path.exists(bot_file):
                log(f"Missing file: {bot_file}", log_file=log_file)
                continue
            X, y, h, w, nb, l2rgb = load_split_npz(bot_file, rgb_json)
        else:
            X, y, h, w, nb, l2rgb = load_image_gdal(ds["feature_path"], ds["label_path"], verbose=1)
        
        y_2d = y.reshape((h, w))
        
        splits = get_spatial_holdout_splits(
            y_2d=y_2d, n_splits=N_SPLITS, hole_size=HOLE_SIZE,
            buffer_size=BUFFER_SIZE, max_trials=MAX_TRIALS, seed=VALIDATION_RANDOM_STATE,
            verbose=1, log_file=log_file
        )
        
        for split_idx, (train_idx, test_idx, holes) in enumerate(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test,  y_test  = X[test_idx],  y[test_idx]
            
            for m_name, model in models.items():
                s_key = f"{dataset_name}__{m_name}__fold{split_idx}"
                if state.get(s_key, False):
                    log(f"  [SKIPPED] Model: {m_name} | Fold {split_idx + 1}/{N_SPLITS} (Already completed)", log_file=log_file)
                    continue
                    
                log(f"  Evaluating Model: {m_name} | Fold {split_idx + 1}/{N_SPLITS}...", log_file=log_file)
                
                try:
                    t0 = time.time()
                    if m_name.startswith("ACC11_"):
                        model.fit(X_train, y_train, spatial_shape=(h, w), train_idx=train_idx)
                    else:
                        model.fit(X_train, y_train)
                    fit_time = time.time() - t0

                    t1 = time.time()
                    if m_name.startswith("ACC11_"):
                        y_pred_full = model.predict(X, spatial_shape=(h, w))
                        predict_time = time.time() - t1
                        y_pred_eval = y_pred_full[test_idx]
                    else:
                        y_pred_eval = model.predict(X_test)
                        predict_time = time.time() - t1

                    y_true_eval = y_test
                    
                    oa = accuracy_score(y_true_eval, y_pred_eval)
                    macro_p = precision_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
                    macro_r = recall_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
                    macro_f1 = f1_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
                    kappa = cohen_kappa_score(y_true_eval, y_pred_eval)
                    mcc = matthews_corrcoef(y_true_eval, y_pred_eval)

                    labels = np.unique(np.concatenate([y_true_eval, y_pred_eval]))
                    labels = np.sort(labels)
                    class_p = precision_score(y_true_eval, y_pred_eval, average=None, labels=labels, zero_division=0)
                    class_r = recall_score(y_true_eval, y_pred_eval, average=None, labels=labels, zero_division=0)
                    class_f1 = f1_score(y_true_eval, y_pred_eval, average=None, labels=labels, zero_division=0)

                    row_data = [dataset_name, m_name, str(split_idx+1), str(oa), str(macro_p), str(macro_r), str(macro_f1), str(kappa), str(mcc), str(fit_time), str(predict_time)]
                    
                    class_metrics = {}
                    for i, cls in enumerate(labels):
                        class_metrics[f"Class_{cls}_Precision"] = str(class_p[i])
                        class_metrics[f"Class_{cls}_Recall"] = str(class_r[i])
                        class_metrics[f"Class_{cls}_F1"] = str(class_f1[i])
                        
                    for cls in range(1, 15):
                        row_data.append(class_metrics.get(f"Class_{cls}_Precision", ""))
                        row_data.append(class_metrics.get(f"Class_{cls}_Recall", ""))
                        row_data.append(class_metrics.get(f"Class_{cls}_F1", ""))
                        row_data.append("") # Threshold is blank

                    with open(csv_path, "a", encoding="utf-8-sig") as f:
                        f.write(",".join(row_data) + "\n")

                    state[s_key] = True
                    with open(state_file, "w") as f:
                        json.dump(state, f)

                except Exception as e:
                    log(f"    [ERROR] {m_name} failed: {e}", log_file=log_file)
                    traceback.print_exc()

if __name__ == "__main__":
    main()
