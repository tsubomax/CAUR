import time
import numpy as np
import scipy.ndimage
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score

class CAURClassifier(BaseEstimator, ClassifierMixin):
    """
    Class-Adaptive Uncertainty Revocation (CAUR) Classifier Framework.
    
    This wrapper applies a lightweight post-classification refinement 
    step to any scikit-learn compatible base estimator. It is designed 
    specifically for improving spatial consistency in remote sensing 
    classification tasks without retraining the base model.
    """
    def __init__(
        self,
        estimator=None,
        diffusion_weight=0.3,
        spatial_threshold=3,
        reconstruction_iters=5,
        percentiles_to_test=(50, 55, 60, 65, 70, 75, 80, 85, 90, 95),
        optimization_metric="macro_f1", 
        random_state=0,
        n_jobs=-1
    ):
        self.estimator = estimator
        self.diffusion_weight = diffusion_weight
        self.spatial_threshold = spatial_threshold
        self.reconstruction_iters = reconstruction_iters
        self.percentiles_to_test = percentiles_to_test
        self.optimization_metric = optimization_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.classes_ = None
        self.class_thresholds_ = {}
        self.base_estimator_ = None
        self.timing_log_ = {}

    def fit(self, X, y, spatial_shape=None, train_idx=None):
        start_total = time.time()
        self.classes_ = np.unique(y)
        
        if self.estimator is None:
            self.base_estimator_ = RandomForestClassifier(
                n_estimators=100, 
                oob_score=True, 
                random_state=self.random_state, 
                n_jobs=self.n_jobs
            )
        else:
            self.base_estimator_ = clone(self.estimator)
            if hasattr(self.base_estimator_, "oob_score"):
                self.base_estimator_.oob_score = True

        # Base Prediction (Training)
        t0 = time.time()
        self.base_estimator_.fit(X, y)
        self.timing_log_["fit_base_model"] = time.time() - t0

        # Optimization phase using OOB or Training predictions
        t0 = time.time()
        use_oob = hasattr(self.base_estimator_, "oob_decision_function_")
        
        if use_oob and spatial_shape is not None and train_idx is not None:
            # OOB Optimization (for Tree-based ensemble models)
            oob_proba = self.base_estimator_.oob_decision_function_
            nan_mask = np.isnan(oob_proba).any(axis=1)
            if np.any(nan_mask):
                oob_proba[nan_mask] = self.base_estimator_.predict_proba(X[nan_mask])

            H, W = spatial_shape
            full_proba = np.zeros((H * W, len(self.classes_)), dtype=np.float32)
            full_proba[train_idx] = oob_proba.astype(np.float32)
            prob_map_2d = full_proba.reshape((H, W, len(self.classes_)))

            y_2d = np.zeros(H * W, dtype=np.int16)
            y_2d[train_idx] = y.astype(np.int16)
            y_2d = y_2d.reshape((H, W))

            train_mask_1d = np.zeros(H * W, dtype=bool)
            train_mask_1d[train_idx] = True
            train_mask = train_mask_1d.reshape((H, W))

            # Probability Diffusion
            if self.diffusion_weight > 0:
                w = self.diffusion_weight
                for k in range(len(self.classes_)):
                    local_mean = scipy.ndimage.uniform_filter(prob_map_2d[:, :, k], size=3, mode="constant", cval=0.0)
                    prob_map_2d[:, :, k] = (1.0 - w) * prob_map_2d[:, :, k] + w * local_mean

            # Entropy Map
            entropy_map = -np.sum(prob_map_2d * np.log(prob_map_2d + 1e-12), axis=2).astype(np.float32)

            # Class-Adaptive Threshold Optimization
            for c in self.classes_:
                c_mask = train_mask & (y_2d == c)
                if np.any(c_mask):
                    self.class_thresholds_[c] = np.percentile(entropy_map[c_mask], 90)
                else:
                    self.class_thresholds_[c] = 999.0

            for c in self.classes_:
                c_mask = train_mask & (y_2d == c)
                if not np.any(c_mask):
                    continue

                best_score = -1.0
                best_thresh = self.class_thresholds_[c]
                candidate_thresholds = np.percentile(entropy_map[c_mask], self.percentiles_to_test)
                initial_pred_2d = self.classes_[np.argmax(prob_map_2d, axis=2)].astype(np.int16)
                
                kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)

                for thresh in candidate_thresholds:
                    current_pred = initial_pred_2d.copy()
                    
                    # Entropy Revocation
                    for cl in self.classes_:
                        th_val = thresh if cl == c else self.class_thresholds_[cl]
                        revoke_mask = (current_pred == cl) & (entropy_map > th_val)
                        current_pred[revoke_mask] = -999

                    # Spatial Consistency Revocation
                    if self.spatial_threshold > 0:
                        for cl in self.classes_:
                            mask = (current_pred == cl)
                            neighbor_counts = scipy.ndimage.convolve(mask.astype(np.uint8), kernel, mode="constant", cval=0)
                            revoke_mask = mask & (neighbor_counts <= self.spatial_threshold)
                            current_pred[revoke_mask] = -999

                    # Fallback Recovery (Skipping reconstruction during fast OOB opt)
                    fallback_mask = (current_pred == -999)
                    current_pred[fallback_mask] = initial_pred_2d[fallback_mask]

                    y_true_eval = y_2d[train_mask]
                    y_pred_eval = current_pred[train_mask]

                    if self.optimization_metric == "macro_f1":
                        score = f1_score(y_true_eval, y_pred_eval, average="macro", zero_division=0)
                    else:
                        score = cohen_kappa_score(y_true_eval, y_pred_eval)

                    if score > best_score:
                        best_score = score
                        best_thresh = thresh
                
                self.class_thresholds_[c] = best_thresh
        else:
            # Fallback estimation for non-OOB classifiers
            prob_X = self.base_estimator_.predict_proba(X).astype(np.float32)
            entropy_X = -np.sum(prob_X * np.log(prob_X + 1e-12), axis=1)
            for c in self.classes_:
                c_mask = (y == c)
                if np.any(c_mask):
                    self.class_thresholds_[c] = np.percentile(entropy_X[c_mask], 90)
                else:
                    self.class_thresholds_[c] = 999.0

        self.timing_log_["optimization"] = time.time() - t0
        self.timing_log_["fit_total"] = time.time() - start_total
        return self

    def predict(self, X, spatial_shape=None):
        start_total = time.time()
        
        if spatial_shape is None:
            # Fallback to standard prediction if no spatial info is provided
            return self.base_estimator_.predict(X)

        H, W = spatial_shape
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)

        # Stage 0: Initial Prediction
        t0 = time.time()
        prob_map_1d = self.base_estimator_.predict_proba(X).astype(np.float32)
        prob_map_2d = prob_map_1d.reshape((H, W, len(self.classes_)))
        initial_pred_2d = self.classes_[np.argmax(prob_map_2d, axis=2)].astype(np.int16)
        self.timing_log_["predict_base"] = time.time() - t0

        # Stage 1: Probability Diffusion
        t0 = time.time()
        if self.diffusion_weight > 0:
            w = self.diffusion_weight
            for k in range(len(self.classes_)):
                local_mean = scipy.ndimage.uniform_filter(prob_map_2d[:, :, k], size=3, mode="constant", cval=0.0)
                prob_map_2d[:, :, k] = (1.0 - w) * prob_map_2d[:, :, k] + w * local_mean
        self.timing_log_["predict_diffusion"] = time.time() - t0

        # Stage 2: Entropy Calculation
        t0 = time.time()
        entropy_map = -np.sum(prob_map_2d * np.log(prob_map_2d + 1e-12), axis=2).astype(np.float32)
        self.timing_log_["predict_entropy_map"] = time.time() - t0

        current_pred = initial_pred_2d.copy()

        # Stage 3A: Uncertainty Revocation
        t0 = time.time()
        for cl in self.classes_:
            th = self.class_thresholds_.get(cl, 999.0)
            revoke_mask = (current_pred == cl) & (entropy_map > th)
            current_pred[revoke_mask] = -999
        self.timing_log_["predict_entropy_revoc"] = time.time() - t0

        # Stage 3B: Spatial Revocation
        t0 = time.time()
        if self.spatial_threshold > 0:
            for cl in self.classes_:
                mask = (current_pred == cl)
                neighbor_counts = scipy.ndimage.convolve(mask.astype(np.uint8), kernel, mode="constant", cval=0)
                revoke_mask = mask & (neighbor_counts <= self.spatial_threshold)
                current_pred[revoke_mask] = -999
        self.timing_log_["predict_spatial_revoc"] = time.time() - t0

        # Stage 4: Iterative Neighborhood Reconstruction
        t0 = time.time()
        if self.reconstruction_iters > 0:
            for iteration in range(self.reconstruction_iters):
                unknown_mask = (current_pred == -999)
                if np.sum(unknown_mask) == 0:
                    break
                
                votes = np.zeros((H, W, len(self.classes_)), dtype=np.int16)
                for c_idx, cl in enumerate(self.classes_):
                    cl_mask = (current_pred == cl).astype(np.int16)
                    votes[:, :, c_idx] = scipy.ndimage.convolve(cl_mask, kernel, mode="constant", cval=0)
                
                max_votes = np.max(votes, axis=2)
                best_class_idx = np.argmax(votes, axis=2)
                
                fill_mask = unknown_mask & (max_votes > 0)
                if np.sum(fill_mask) == 0:
                    break
                
                current_pred[fill_mask] = self.classes_[best_class_idx[fill_mask]]
        self.timing_log_["predict_reconstruction"] = time.time() - t0

        # Fallback Recovery
        t0 = time.time()
        fallback_mask = (current_pred == -999)
        current_pred[fallback_mask] = initial_pred_2d[fallback_mask]
        self.timing_log_["predict_fallback"] = time.time() - t0

        self.timing_log_["predict_total"] = time.time() - start_total
        return current_pred.flatten()

    def predict_proba(self, X):
        return self.base_estimator_.predict_proba(X)
