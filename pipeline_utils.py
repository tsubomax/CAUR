# -*- coding: utf-8 -*-
"""
pipeline_utils.py
パイプライン全体で使うユーティリティ関数群。
ログ出力・進捗管理（レジューム）・計測・評価指標の計算などを担当。
"""

import csv
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.utils.multiclass import unique_labels

# ============================================================
# ログユーティリティ
# ============================================================

def log(message: str, log_file: str = None, verbose: int = 1, level: int = 1, log_only: bool = False):
    """
    コンソールとファイルに同時ログ出力。
    level <= verbose の場合のみコンソールに出力する。
    log_only=True の場合はコンソール出力をスキップしてファイルにのみ書く。
    level=0 : 常に表示（エラー・フェーズ開始終了など最重要）
    level=1 : 通常の進捗表示（Fold完了, モデル開始など）
    level=2 : 詳細（評価値の詳細など）
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    if not log_only and level <= verbose:
        print(full_msg)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
        except IOError as e:
            print(f"[LOG ERROR] {e}")


def log_file_only(message: str, log_file: str):
    """ログファイルにのみ書き込む（ターミナルには出力しない）。
    DEBUGログ・警告など、詳細すぎてターミナルを汚すメッセージ専用。"""
    if not log_file:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except IOError:
        pass


def log_section(title: str, log_file: str = None, verbose: int = 1):
    """セクション区切りを表示。"""
    sep = "=" * 60
    log(sep, log_file, verbose, level=0)
    log(f"  {title}", log_file, verbose, level=0)
    log(sep, log_file, verbose, level=0)


def log_fold(fold_idx: int, n_folds: int, model_name: str, metric_val: float,
             metric_name: str, elapsed: float, log_file: str, verbose: int):
    """1 Foldの結果を簡潔に1行でログ出力。"""
    log(
        f"  [Fold {fold_idx+1}/{n_folds}] Model={model_name} | "
        f"{metric_name}={metric_val:.4f} | Time={elapsed:.1f}s",
        log_file, verbose, level=1
    )


# ============================================================
# 進捗管理（レジューム機能）
# ============================================================

def load_state(state_file: str) -> dict:
    """execution_state.json を読み込んで進捗辞書を返す。なければ空辞書。"""
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state_file: str, state: dict):
    """進捗辞書をJSONに保存する（アトミックに書き込み）。"""
    tmp_file = state_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, state_file)


def is_done(state: dict, key: str) -> bool:
    """指定キーが "DONE" か確認。"""
    return state.get(key) == "DONE"


def mark_done(state_file: str, state: dict, key: str):
    """指定キーを "DONE" にして保存。"""
    state[key] = "DONE"
    save_state(state_file, state)


def mark_skipped(state_file: str, state: dict, key: str, reason: str = ""):
    """指定キーを "SKIPPED" にして保存。"""
    state[key] = f"SKIPPED:{reason}"
    save_state(state_file, state)


# ============================================================
# 実行時間計測
# ============================================================

class Timer:
    """with ブロックで使える簡易タイマー。
    __enter__ 時に「>>> 実行中: ラベル」をターミナルに表示する。
    """
    def __init__(self, label: str = "", time_log_file: str = None, verbose: int = 1, level: int = 1):
        self.label = label
        self.time_log_file = time_log_file
        self.verbose = verbose
        self.level = level
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        # 開始時にターミナルへ「何を実行中か」を表示（level=0 で常時表示）
        start_msg = f">>> 実行中: {self.label}"
        print(start_msg, flush=True)
        if self.time_log_file:
            try:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.time_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] [START] {self.label}\n")
            except IOError:
                pass
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        msg = f"[TIME] {self.label}: {self.elapsed:.2f}s ({self.elapsed/60:.2f}min)"
        log(msg, None, self.verbose, level=self.level)
        if self.time_log_file:
            try:
                with open(self.time_log_file, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
            except IOError:
                pass


# ============================================================
# 画像読み込み・分割・保存
# ============================================================

def load_image_gdal(feature_path: str, label_path: str, verbose: int = 1):
    """
    rasterioを使って特徴量画像とラベル画像を読み込み、
    (X_flat, y_flat, height, width, n_bands, label_to_rgb_map) を返す。
    X_flat: (H*W, bands), y_flat: (H*W,)
    """
    import rasterio
    import numpy as np

    # --- ラベル読み込み ---
    with rasterio.open(label_path) as src_label:
        height = src_label.height
        width = src_label.width
        bands = src_label.count
        if bands == 3:
            rgb_arr = np.transpose(src_label.read(), (1, 2, 0))
            y_2d, rgb_to_label, label_to_rgb = _generate_labels_from_rgb(rgb_arr)
        elif bands == 1:
            y_2d = src_label.read(1)
            unique_l = np.unique(y_2d)
            max_l = max(unique_l) if max(unique_l) > 0 else 1
            label_to_rgb = {int(l): (int(l*255//max_l),)*3 for l in unique_l}
        else:
            raise ValueError(f"Unsupported label band count: {bands}")

    # --- 特徴量読み込み ---
    feat_open = _resolve_feature_path(feature_path)
    with rasterio.open(feat_open) as src_feat:
        if src_feat.width != width or src_feat.height != height:
            raise ValueError(
                f"Dimension mismatch: label=({width}x{height}), "
                f"feature=({src_feat.width}x{src_feat.height})"
            )
        n_bands = src_feat.count
        X_raw = src_feat.read()   # (bands, H, W)

    # (bands, H, W) → (H, W, bands) → (H*W, bands)
    X_flat = np.transpose(X_raw, (1, 2, 0)).reshape(-1, n_bands).astype(np.float32)
    y_flat = y_2d.flatten()

    # NaN/Inf 除去
    if not np.all(np.isfinite(X_flat)):
        X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

    return X_flat, y_flat, height, width, n_bands, label_to_rgb


def split_image_data(X_flat, y_flat, height, width, split_ratio: float):
    """
    フラット化されたピクセルデータを「上部 split_ratio」と「下部 (1-split_ratio)」に分割する。
    画像行(Y)ベースで切り出すため、ガタつきが生じない（横一直線の分割）。
    Returns:
        (X_top, y_top, X_bot, y_bot, split_idx)
    """
    split_idx = int(height * split_ratio)
    top_end = split_idx * width
    X_top = X_flat[:top_end]
    y_top = y_flat[:top_end]
    X_bot = X_flat[top_end:]
    y_bot = y_flat[top_end:]
    return X_top, y_top, X_bot, y_bot, split_idx


def save_split_npz(npz_path: str, rgb_json_path: str,
                   X_flat, y_flat, height, width, n_bands, label_to_rgb,
                   verbose: int = 1):
    """分割データをNPZ形式で保存する。"""
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        X=X_flat, y=y_flat,
        height=np.array([height]),
        width=np.array([width]),
        n_bands=np.array([n_bands])
    )
    with open(rgb_json_path, "w") as f:
        json.dump({str(k): [int(x) for x in v] for k, v in label_to_rgb.items()}, f)
    if verbose >= 1:
        print(f"  Saved split data: {npz_path} (X={X_flat.shape}, y={y_flat.shape})")
    return npz_path


def load_split_npz(npz_path: str, rgb_json_path: str = None):
    """
    save_split_npz で保存したNPZを読み込む。
    Returns: (X, y, height, width, n_bands, label_to_rgb)
    """
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    height  = int(data["height"][0])
    width   = int(data["width"][0])
    n_bands = int(data["n_bands"][0])
    label_to_rgb = {}
    if rgb_json_path and os.path.exists(rgb_json_path):
        with open(rgb_json_path, "r") as f:
            raw = json.load(f)
        label_to_rgb = {int(k): tuple(v) for k, v in raw.items()}
    return X, y, height, width, n_bands, label_to_rgb


# ============================================================
# 評価関数
# ============================================================

def evaluate_model(
    model_name: str,
    model_instance,
    X_train, y_train,
    X_test, y_test,
    eval_labels=None,
):
    """
    モデルを学習・評価し結果辞書を返す。
    Macro F1は sklearn の標準定義（背景クラス0を除外したラベルリストを labels= に渡す）を使用。
    eval_labels 引数は外部から渡してもここでは使用せず、y_trainの非背景ラベルを使う。
    """
    results = {
        "accuracy": 0.0, "mcc": 0.0,
        "macro_f1": 0.0, "weighted_f1": 0.0,
        "macro_precision": 0.0, "macro_recall": 0.0,
        "unclassified_rate": 0.0, "unknown_rate": 0.0,
        "fit_time": -1.0, "pred_time": -1.0,
        "confusion_matrix": np.array([]),
        "report_str": "N/A",
        "error_info": None,
    }

    try:
        # --- 学習 ---
        t0 = time.time()
        if model_name == "TabNet":
            # TabNet は Early Stopping のために eval_set を要求する
            model_instance.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=["accuracy"],
                patience=5,
                max_epochs=100
            )
        else:
            model_instance.fit(X_train, y_train)
        results["fit_time"] = time.time() - t0

        # --- 予測 ---
        t1 = time.time()
        y_pred = model_instance.predict(X_test)
        results["pred_time"] = time.time() - t1

        # --- 未分類率（ACC用） ---
        is_unclassified = np.zeros(len(y_pred), dtype=bool)
        if hasattr(model_instance, "_is_unclassified"):
            try:
                is_unclassified = model_instance._is_unclassified(y_pred)
            except Exception:
                pass
        results["unclassified_rate"] = float(np.mean(is_unclassified))

        # 未分類を除いたサンプルで評価
        classified_mask = ~is_unclassified
        y_test_cls  = y_test[classified_mask]
        y_pred_cls  = y_pred[classified_mask]

        if len(y_test_cls) == 0:
            results["error_info"] = "No classified samples."
            return results

        # --- 未知クラス率（訓練に存在しないクラスがテストに出現） ---
        known_nz = sorted([l for l in unique_labels(y_train) if l != 0])
        unknown_mask = (~np.isin(y_test_cls, known_nz)) & (y_test_cls != 0)
        results["unknown_rate"] = float(np.mean(unknown_mask))

        # --- Accuracy, MCC ---
        results["accuracy"] = float(accuracy_score(y_test_cls, y_pred_cls))
        try:
            results["mcc"] = float(matthews_corrcoef(y_test_cls, y_pred_cls))
        except Exception:
            results["mcc"] = 0.0

        # --- Macro F1 (sklearn 標準定義、背景クラス0を除外) ---
        # 評価対象ラベル：訓練時に存在した非背景クラス
        used_labels = known_nz if known_nz else sorted(set(y_test_cls.tolist()))
        if not used_labels:
            used_labels = None  # 全クラスで計算

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report_dict = classification_report(
                y_test_cls, y_pred_cls,
                labels=used_labels,
                zero_division=0,
                output_dict=True,
            )
            report_str = classification_report(
                y_test_cls, y_pred_cls,
                labels=used_labels,
                zero_division=0,
            )

        if "macro avg" in report_dict:
            results["macro_f1"]        = float(report_dict["macro avg"].get("f1-score", 0.0))
            results["macro_precision"] = float(report_dict["macro avg"].get("precision", 0.0))
            results["macro_recall"]    = float(report_dict["macro avg"].get("recall", 0.0))
        if "weighted avg" in report_dict:
            results["weighted_f1"] = float(report_dict["weighted avg"].get("f1-score", 0.0))

        # クラス別指標の抽出（CSV出力用）
        results["class_metrics"] = {}
        if used_labels:
            for cls_lbl in used_labels:
                cls_str = str(cls_lbl)
                if cls_str in report_dict:
                    results["class_metrics"][cls_lbl] = {
                        "precision": float(report_dict[cls_str].get("precision", 0.0)),
                        "recall":    float(report_dict[cls_str].get("recall", 0.0)),
                        "f1":        float(report_dict[cls_str].get("f1-score", 0.0)),
                        "support":   int(report_dict[cls_str].get("support", 0))
                    }

        results["report_str"] = report_str

        # --- 混同行列 ---
        if used_labels:
            results["confusion_matrix"] = confusion_matrix(
                y_test_cls, y_pred_cls, labels=used_labels
            )

    except Exception as e:
        results["error_info"] = f"{e}\n{traceback.format_exc()}"

    return results


# ============================================================
# CSV出力
# ============================================================

CSV_HEADER = [
    "Phase", "Dataset", "Fold", "Model_Name",
    "Accuracy", "MCC", "Macro_F1", "Weighted_F1",
    "Macro_Precision", "Macro_Recall",
    "Unclassified_Rate", "Unknown_Rate",
    "Fit_Time_s", "Pred_Time_s",
    "Error_Info",
]

CLASS_CSV_HEADER = [
    "Phase", "Dataset", "Fold", "Model_Name", "Class_Label",
    "Precision", "Recall", "F1_Score", "Support"
]


def init_csv(csv_path: str):
    """CSVファイルを初期化（ヘッダー書き込み）。既に存在する場合は追記のみ。"""
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(CSV_HEADER)


def init_class_csv(csv_path: str):
    """クラス別CSVファイルを初期化。"""
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(CLASS_CSV_HEADER)


def append_csv_row(csv_path: str, phase: str, dataset: str, fold: int,
                   model_name: str, results: dict):
    """結果辞書から1行CSVに追記。"""
    def fmt(v, decimals=4):
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        if v is None:
            return "N/A"
        return str(v)

    row = [
        phase, dataset, fold, model_name,
        fmt(results.get("accuracy")),
        fmt(results.get("mcc")),
        fmt(results.get("macro_f1")),
        fmt(results.get("weighted_f1")),
        fmt(results.get("macro_precision")),
        fmt(results.get("macro_recall")),
        fmt(results.get("unclassified_rate")),
        fmt(results.get("unknown_rate")),
        fmt(results.get("fit_time")),
        fmt(results.get("pred_time")),
        str(results.get("error_info") or ""),
    ]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row)


def append_class_csv_row(csv_path: str, phase: str, dataset: str, fold: int,
                         model_name: str, results: dict):
    """クラス別の結果をCSVに追記。"""
    if "class_metrics" not in results or not results["class_metrics"]:
        return

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for cls_lbl, metrics in results["class_metrics"].items():
            row = [
                phase, dataset, fold, model_name, cls_lbl,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                str(metrics['support'])
            ]
            writer.writerow(row)


# ============================================================
# 統計検定
# ============================================================

def run_statistical_tests(scores_by_model: dict, baseline_model: str,
                           log_file: str, verbose: int = 1):
    """
    baseline_model のFold別スコアと各モデルのFold別スコアに対して
    ウィルコクソン符号順位検定（またはt検定）を実施し、結果をログに出力。

    Args:
        scores_by_model: {"ModelA": [f1_fold1, f1_fold2, ...], ...}
        baseline_model: ベースライン名（比較基準）
        log_file: ログファイルパス
        verbose: 表示レベル
    Returns:
        dict: {"ModelA": {"statistic": ..., "pvalue": ..., "test": ...}, ...}
    """
    from scipy import stats

    log_section("Statistical Tests (vs Baseline: " + baseline_model + ")", log_file, verbose)

    baseline_scores = scores_by_model.get(baseline_model)
    if baseline_scores is None:
        log(f"  [WARN] Baseline model '{baseline_model}' not found in scores.", log_file, verbose)
        return {}

    results = {}
    for model_name, model_scores in scores_by_model.items():
        if model_name == baseline_model:
            continue
        if len(model_scores) != len(baseline_scores):
            log(f"  [WARN] Fold count mismatch for {model_name}. Skipping.", log_file, verbose)
            continue

        diff = np.array(model_scores) - np.array(baseline_scores)
        n = len(diff)
        test_name = ""
        stat, pval = None, None

        # ウィルコクソン検定は n>=3 必要、それ以下はt検定
        if n >= 3 and not np.all(diff == 0):
            try:
                stat, pval = stats.wilcoxon(diff)
                test_name = "Wilcoxon"
            except Exception:
                stat, pval = stats.ttest_rel(baseline_scores, model_scores)
                test_name = "paired-t"
        else:
            try:
                stat, pval = stats.ttest_rel(baseline_scores, model_scores)
                test_name = "paired-t"
            except Exception:
                stat, pval = 0.0, 1.0
                test_name = "N/A"

        sig = "★SIGNIFICANT★" if pval is not None and pval < 0.05 else ""
        log(
            f"  {model_name} vs {baseline_model}: "
            f"test={test_name}, stat={stat:.4f}, p={pval:.4f} {sig}",
            log_file, verbose
        )
        results[model_name] = {"test": test_name, "statistic": float(stat), "pvalue": float(pval)}

    return results


# ============================================================
# 予測画像保存
# ============================================================

def save_prediction_image(y_pred, height, width, label_to_rgb, out_path: str):
    """予測ラベル配列をRGB TIFで保存。"""
    try:
        from PIL import Image
        pred_2d = y_pred.reshape(height, width)
        rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
        for lbl, rgb in label_to_rgb.items():
            mask = pred_2d == lbl
            rgb_img[mask] = np.array(rgb[:3], dtype=np.uint8)
        Image.fromarray(rgb_img, "RGB").save(out_path, format="TIFF", compression="tiff_lzw")
    except Exception as e:
        print(f"  [WARN] Could not save prediction image: {e}")


# ============================================================
# 内部ヘルパー
# ============================================================

def _generate_labels_from_rgb(rgb_array):
    """RGB配列 (H,W,3) から一意なラベルを生成。"""
    h, w, _ = rgb_array.shape
    reshaped = rgb_array.reshape(-1, 3)
    unique_colors, indices = np.unique(reshaped, axis=0, return_inverse=True)
    rgb_to_label = {tuple(c): i for i, c in enumerate(unique_colors)}
    label_to_rgb = {i: tuple(c) for i, c in enumerate(unique_colors)}
    label_image  = indices.reshape(h, w)
    return label_image, rgb_to_label, label_to_rgb


def _resolve_feature_path(path: str) -> str:
    """HDRファイルが指定された場合、対応するデータファイルパスに変換。"""
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() == ".hdr":
        for ext in ["", ".img", ".dat", ".bip", ".bsq"]:
            candidate = p.parent / (p.stem + ext)
            if candidate.is_file():
                return str(candidate)
        raise IOError(f"Cannot find data file for HDR: {path}")
    return path

# ============================================================
# Spatial Holdout Utils
# ============================================================

def get_spatial_holdout_splits(y_2d: np.ndarray, n_splits: int, hole_size: int, buffer_size: int,
                               max_trials: int, seed: int, verbose: int = 1, log_file: str = None):
    np.random.seed(seed)
    h, w = y_2d.shape
    unique_classes = [c for c in np.unique(y_2d) if c != 0]
    total_labeled_pixels = np.sum(y_2d != 0)
    total_counts = {c: np.sum(y_2d == c) for c in unique_classes}
    
    splits = []
    
    for split_idx in range(n_splits):
        found = False
        log(f"  -> Sampling Double Spatial Hole for Fold {split_idx+1}/{n_splits}...", log_file, verbose, level=2)
        
        for trial in range(max_trials):
            r1 = np.random.randint(0, h - (hole_size + 2 * buffer_size) + 1)
            c1 = np.random.randint(0, w - (hole_size + 2 * buffer_size) + 1)
            
            r2 = np.random.randint(0, h - (hole_size + 2 * buffer_size) + 1)
            c2 = np.random.randint(0, w - (hole_size + 2 * buffer_size) + 1)
            
            if abs(r1 - r2) < hole_size and abs(c1 - c2) < hole_size:
                continue
            
            hole1_y = y_2d[r1+buffer_size:r1+buffer_size+hole_size, c1+buffer_size:c1+buffer_size+hole_size]
            hole2_y = y_2d[r2+buffer_size:r2+buffer_size+hole_size, c2+buffer_size:c2+buffer_size+hole_size]
            
            classes_in_holes = set(np.unique(hole1_y)).union(set(np.unique(hole2_y)))
            classes_in_holes.discard(0)
            
            if len(classes_in_holes) < len(unique_classes):
                continue
            
            mask_train = np.ones_like(y_2d, dtype=bool)
            mask_train[r1:r1+hole_size+2*buffer_size, c1:c1+hole_size+2*buffer_size] = False
            mask_train[r2:r2+hole_size+2*buffer_size, c2:c2+hole_size+2*buffer_size] = False
            
            train_y = y_2d[mask_train]
            
            valid = True
            for cls in unique_classes:
                count_in_train = np.sum(train_y == cls)
                if count_in_train / total_counts[cls] < 0.8:
                    valid = False
                    break
            
            if not valid:
                continue
                
            mask_test = np.zeros_like(y_2d, dtype=bool)
            mask_test[r1+buffer_size:r1+buffer_size+hole_size, c1+buffer_size:c1+buffer_size+hole_size] = True
            mask_test[r2+buffer_size:r2+buffer_size+hole_size, c2+buffer_size:c2+buffer_size+hole_size] = True
            
            train_idx = np.where(mask_train.flatten())[0]
            test_idx = np.where(mask_test.flatten())[0]
            
            holes = [(r1, c1), (r2, c2)]
            splits.append((train_idx, test_idx, holes))
            found = True
            break
            
        if not found:
            log(f"  [WARN] Could not find double holes for fold {split_idx+1}!", log_file, verbose)
            splits.append((None, None, []))
            
    return splits

def save_spatial_split_image(y_2d: np.ndarray, holes: list, hole_size: int, buffer_size: int, out_path: str):
    try:
        import os
        from PIL import Image
        h, w = y_2d.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        valid_mask = (y_2d != 0)
        rgb_img[valid_mask] = [255, 255, 255]
        
        buffer_mask = np.zeros((h, w), dtype=bool)
        test_mask = np.zeros((h, w), dtype=bool)
        
        for (r, c) in holes:
            buffer_mask[r:r + hole_size + 2 * buffer_size, c:c + hole_size + 2 * buffer_size] = True
            hole_r_start, hole_r_end = r + buffer_size, r + buffer_size + hole_size
            hole_c_start, hole_c_end = c + buffer_size, c + buffer_size + hole_size
            test_mask[hole_r_start:hole_r_end, hole_c_start:hole_c_end] = True
        
        buffer_valid = buffer_mask & valid_mask
        rgb_img[buffer_valid] = [128, 128, 128]
        
        test_valid = test_mask & valid_mask
        rgb_img[test_valid] = [255, 0, 0]
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(rgb_img, "RGB").save(out_path, format="PNG")
    except Exception as e:
        print(f"  [WARN] Could not save spatial split image: {e}")

def append_summary_rows(csv_path: str, phase: str, dataset: str, model_name: str, results_list: list):
    """結果辞書のリストからMeanとStdを計算してCSVに追記する。"""
    if not results_list:
        return
        
    keys_to_agg = [
        "accuracy", "mcc", "macro_f1", "weighted_f1", 
        "macro_precision", "macro_recall", 
        "unclassified_rate", "unknown_rate",
        "fit_time", "pred_time"
    ]
    
    mean_res = {}
    std_res = {}
    
    for k in keys_to_agg:
        vals = [r.get(k, 0.0) for r in results_list if isinstance(r.get(k), (int, float))]
        if vals:
            mean_res[k] = float(np.mean(vals))
            std_res[k] = float(np.std(vals))
        else:
            mean_res[k] = None
            std_res[k] = None
            
    mean_res["error_info"] = ""
    std_res["error_info"] = ""
    
    append_csv_row(csv_path, phase, dataset, "Mean", model_name, mean_res)
    append_csv_row(csv_path, phase, dataset, "Std", model_name, std_res)

