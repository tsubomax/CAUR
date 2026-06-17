# -*- coding: utf-8 -*-
"""
pipeline_config.py
繝代う繝励Λ繧､繝ｳ蜈ｨ菴薙・險ｭ螳夲ｼ医ヱ繧ｹ繝ｻ繝代Λ繝｡繝ｼ繧ｿ・峨ｒ荳蜈・ｮ｡逅・☆繧玖ｨｭ螳壹ヵ繧｡繧､繝ｫ縲・
"""

import os

# ============================================================
# GPU螳悟・辟｡蜉ｹ蛹厄ｼ・UDA_VISIBLE_DEVICES 繧堤ｩｺ縺ｫ縺吶ｋ縺薙→縺ｧ
# PyTorch縺悟ｮ溯｡梧凾縺ｫGPU繧定ｪ崎ｭ倥＠縺ｪ縺上↑繧翫€，UDA蛻晄悄蛹冶ｭｦ蜻翫′豸医∴繧具ｼ・
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ============================================================
# 蜃ｺ蜉帙ヱ繧ｹ險ｭ螳・
# ============================================================
BASE_OUTPUT_DIR      = r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/Validation"
SPLIT_DATA_DIR       = os.path.join(r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/OptunaVer5", "DATA", "Cuprite_Split")
CACHE_DIR            = os.path.join(BASE_OUTPUT_DIR, "DATA", "Resampled_MS_Cache")
RESULTS_DIR          = os.path.join(BASE_OUTPUT_DIR, "results")
IMAGES_DIR           = os.path.join(BASE_OUTPUT_DIR, "pred_images")

STATE_FILE           = os.path.join(BASE_OUTPUT_DIR, "acc11_execution_state.json")
TIME_LOG_FILE        = os.path.join(BASE_OUTPUT_DIR, "acc11_execution_time.log")

# Optuna繧ｹ繧ｿ繝・ぅ菫晏ｭ伜・・医Ο繝・け繧ｨ繝ｩ繝ｼ蝗樣∩縺ｮ縺溘ａ繝ｭ繝ｼ繧ｫ繝ｫ縺ｮ/tmp繧剃ｽｿ逕ｨ・・
OPTUNA_DB_PATH       = "/tmp/optuna_ver5.db"

# ============================================================
# 蛻・牡險ｭ螳・
# ============================================================
SPLIT_RATIO          = 0.3      # 荳企Κ縺ｨ荳矩Κ縺ｮ蛻・牡豈皮紫
TOP_PCT              = int(SPLIT_RATIO * 100)
BOT_PCT              = 100 - TOP_PCT
TOP_NAME             = f"top{TOP_PCT}"
BOT_NAME             = f"bot{BOT_PCT}"

# ============================================================
# 繝・・繧ｿ繧ｻ繝・ヨ螳夂ｾｩ (繝壹い繝ｪ繝ｳ繧ｰ縺ｮ蝗ｺ螳・
# ============================================================
CUPRITE_MS = r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/MS/Cuprite_aster"
MITO_MS    = r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/MS/Mito_aster"

# 繝√Η繝ｼ繝九Φ繧ｰ逕ｨ繝・・繧ｿ・・hase 1・・
TUNING_DATASET = {
    "name": "Cuprite_AVIRIS_Hourglass",
    "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/cuprite/AVIRIS_HG.tif",
    "feature_path": CUPRITE_MS,
    "is_cuprite": True
}

# 譛ｬ逡ｪ讀懆ｨｼ逕ｨ繝・・繧ｿ・・hase 2縲・・・
VALIDATION_DATASETS = [
    # Cuprite 5繝代ち繝ｼ繝ｳ
    {"name": "Cuprite_AVIRIS_Hourglass",   "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/cuprite/AVIRIS_HG.tif", "feature_path": CUPRITE_MS, "is_cuprite": True},
    {"name": "Cuprite_AVIRIS_Tetracorder", "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/cuprite/AVIRIS_TC.tif", "feature_path": CUPRITE_MS, "is_cuprite": True},
    {"name": "Cuprite_EMIT_Hourglass",     "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/cuprite/EMIT_HG.tif",  "feature_path": CUPRITE_MS, "is_cuprite": True},
    {"name": "Cuprite_EMIT_Tetracorder",   "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/cuprite/EMIT_TC.tif",  "feature_path": CUPRITE_MS, "is_cuprite": True},
    {"name": "Cuprite_HISUI_Hourglass",    "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/cuprite/HISUI_HG.tif", "feature_path": CUPRITE_MS, "is_cuprite": True},

    # Mito 2繝代ち繝ｼ繝ｳ
    {"name": "Mito_EMIT_Hourglass",        "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/mito/MT_EMIT_HG.tif",  "feature_path": MITO_MS,    "is_cuprite": False},
    {"name": "Mito_HISUI_Class12",         "label_path": r"/share_win/tsubo/Satellite_Image/2026/MDPIAppSci/HS/mito/MT_HISUI_In.tif", "feature_path": MITO_MS,    "is_cuprite": False},
]

# ============================================================
# 繝輔ぉ繝ｼ繧ｺ1・壹ワ繧､繝代・繝代Λ繝｡繝ｼ繧ｿ隱ｿ謨ｴ險ｭ螳・(Optuna逕ｨ)
# ============================================================
TUNING_N_SPLITS      = 5        # 5-Fold
TUNING_N_REPEATS     = 1        # 謗｢邏｢蝗樊焚縺悟､壹＞縺ｮ縺ｧ蝓ｺ譛ｬ1蝗・
TUNING_RANDOM_STATE  = 0
OPTUNA_N_TRIALS      = 20       # 1繝｢繝・Ν縺ゅ◆繧翫・繝吶う繧ｺ謗｢邏｢蝗樊焚

OPTUNA_SEARCH_SPACE = {
    "RandomForest": {
        "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
        "max_depth": {"type": "categorical", "choices": [10, 20, 30, None]},
        "min_samples_split": {"type": "int", "low": 2, "high": 10},
    },
    "LinearSVC": {
        "C": {"type": "loguniform", "low": 0.01, "high": 100.0},
        "max_iter": {"type": "categorical", "choices": [2000]},
    },
    "LogisticRegression": {
        "C": {"type": "loguniform", "low": 0.01, "high": 100.0},
        "max_iter": {"type": "categorical", "choices": [1000]},
        "solver": {"type": "categorical", "choices": ["lbfgs", "saga"]},
    },
    "MLP": {
        "hidden_layer_sizes": {"type": "categorical", "choices": ["(100,)", "(100, 50)", "(200, 100)"]},
        "alpha": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "learning_rate_init": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
    },
    "GaussianNB": {
        "var_smoothing": {"type": "loguniform", "low": 1e-11, "high": 1e-7},
    },
    "KNeighbors": {
        "n_neighbors": {"type": "int", "low": 3, "high": 15},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
    },
    "TabNet": {
        "n_d": {"type": "categorical", "choices": [8, 16, 24]},
        "n_steps": {"type": "int", "low": 3, "high": 7},
        "gamma": {"type": "float", "low": 1.0, "high": 2.0},
    },
    "1D-CNN": {
        "filters": {"type": "categorical", "choices": [32, 64, 128]},
        "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
        "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    },
    "ACC": {
        "unclassified_tolerance_p": {"type": "float", "low": 0.05, "high": 0.3},
        "max_updates": {"type": "int", "low": 1, "high": 3},
        "min_f1_threshold": {"type": "float", "low": 0.70, "high": 0.95},
    }
}

# ============================================================
# 繝輔ぉ繝ｼ繧ｺ2・壽怙邨よ､懆ｨｼ險ｭ螳・
# ============================================================
VALIDATION_N_SPLITS     = 4
VALIDATION_RANDOM_STATE = 0

HOLDOUT_HOLE_SIZE       = 60
HOLDOUT_BUFFER_SIZE     = 2
HOLDOUT_MAX_TRIALS      = 1000

BASELINE_MODEL       = "RandomForest"

# ============================================================
# 蜈ｱ騾夊ｨｭ螳・
# ============================================================
RANDOM_STATE         = 0
N_JOBS               = -1
VERBOSE              = 1
SAVE_PRED_IMAGES     = True

# ============================================================
# 繝輔ぉ繝ｼ繧ｺ4・哂CC・倶ｻ悶Δ繝・Ν險ｭ螳・
# ============================================================
ACC_EXTRA_ESTIMATORS = [
    """
    "LinearSVC",
    "LogisticRegression",
    "MLP",
    "GaussianNB",
    "KNeighbors",
    "TabNet",
    "1D-CNN"
    """
]

ACC_ABLATION_CONFIGS = {
    "ACC_RF_No_OvR": {
        "unclassified_tolerance_p": 1.0,
        "max_updates": 2,
        "feature_generator": "scaler",
    },
    "ACC_RF_No_FeatureTransform": {
        "unclassified_tolerance_p": 0.1,
        "max_updates": 2,
        "feature_generator": None,
    },
    "ACC_RF_SimpleCascade": {
        "unclassified_tolerance_p": 1.0,
        "max_updates": 0,
        "feature_generator": None,
    }
}
