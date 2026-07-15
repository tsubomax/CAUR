"""
Example usage of the CAUR framework with Optuna hyperparameter optimization.
"""
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from caur import CAURClassifier

# --- 1. Generate Synthetic Data for Demonstration ---
def generate_spatial_data(H=50, W=50, n_classes=4, random_state=42):
    np.random.seed(random_state)
    # Create smooth spatial class blocks
    y_2d = np.zeros((H, W), dtype=int)
    for c in range(n_classes):
        cy, cx = np.random.randint(0, H), np.random.randint(0, W)
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        y_2d[dist < 15] = c
    
    # Generate noisy features
    X = np.random.randn(H * W, 10)
    y = y_2d.flatten()
    X += y[:, None] * 0.5  # Add signal
    return X, y, (H, W)

X_data, y_data, spatial_shape = generate_spatial_data()
H, W = spatial_shape
train_idx = np.random.choice(H*W, size=int(0.6 * H*W), replace=False)
test_idx = np.setdiff1d(np.arange(H*W), train_idx)

# --- 2. Define the Optuna Objective Function ---
def objective(trial):
    # Suggest CAUR hyper-parameters
    diffusion_weight = trial.suggest_float("diffusion_weight", 0.0, 0.5, step=0.1)
    spatial_threshold = trial.suggest_int("spatial_threshold", 0, 5)
    reconstruction_iters = trial.suggest_int("reconstruction_iters", 1, 10)
    
    # Suggest Base Classifier parameters
    n_estimators = trial.suggest_categorical("n_estimators", [50, 100])
    
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        oob_score=True, 
        random_state=42, 
        n_jobs=-1
    )
    
    model = CAURClassifier(
        estimator=base_rf,
        diffusion_weight=diffusion_weight,
        spatial_threshold=spatial_threshold,
        reconstruction_iters=reconstruction_iters,
        random_state=42
    )
    
    # Fit the model. CAUR handles OOB inner optimization automatically
    # since we pass spatial_shape and train_idx.
    model.fit(X_data[train_idx], y_data[train_idx], 
              spatial_shape=spatial_shape, train_idx=train_idx)
    
    # Predict and evaluate on the test set
    y_pred = model.predict(X_data, spatial_shape=spatial_shape)
    y_pred_test = y_pred[test_idx]
    
    macro_f1 = f1_score(y_data[test_idx], y_pred_test, average="macro", zero_division=0)
    return macro_f1

# --- 3. Run Optimization ---
if __name__ == "__main__":
    print("Starting Optuna optimization for CAUR...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("\nBest Trial:")
    print("  Macro-F1:", study.best_value)
    print("  Params:", study.best_params)
    
    # Train the final model using best params
    print("\nTraining final CAUR model with best parameters...")
    final_model = CAURClassifier(
        estimator=RandomForestClassifier(
            n_estimators=study.best_params["n_estimators"], 
            oob_score=True, random_state=42
        ),
        diffusion_weight=study.best_params["diffusion_weight"],
        spatial_threshold=study.best_params["spatial_threshold"],
        reconstruction_iters=study.best_params["reconstruction_iters"],
        random_state=42
    )
    final_model.fit(X_data[train_idx], y_data[train_idx], 
                    spatial_shape=spatial_shape, train_idx=train_idx)
    
    y_pred = final_model.predict(X_data, spatial_shape=spatial_shape)
    final_f1 = f1_score(y_data[test_idx], y_pred[test_idx], average="macro", zero_division=0)
    print(f"Final Model Test Macro-F1: {final_f1:.4f}")
