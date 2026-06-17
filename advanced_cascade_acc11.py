import numpy as np
import scipy.ndimage
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score
import time

class ACC11AdaptiveSpatialEntropyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimator=None,
        use_diffusion=True,
        use_adaptive_threshold=True,
        use_entropy_revocation=True,
        use_spatial_revocation=True,
        use_reconstruction=True,
        optimization_metric="macro_f1", # "macro_f1" or "kappa"
        random_state=0,
        n_jobs=-1
    ):
        self.estimator = estimator
        self.use_diffusion = use_diffusion
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_entropy_revocation = use_entropy_revocation
        self.use_spatial_revocation = use_spatial_revocation
        self.use_reconstruction = use_reconstruction
        self.optimization_metric = optimization_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.classes_ = None
        self.class_thresholds_ = {}
        self.base_estimator_ = None
        
        # Logging variables for tracking the processing counts
        self.revoked_pixels_entropy_ = 0
        self.revoked_pixels_spatial_ = 0
        self.reconstructed_pixels_ = 0
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
            # Ensure OOB is enabled if it's a RandomForest
            if hasattr(self.base_estimator_, "oob_score"):
                self.base_estimator_.oob_score = True

        # STEP-A: Base Prediction (Training)
        t0 = time.time()
        self.base_estimator_.fit(X, y)
        self.timing_log_["fit_base_model"] = time.time() - t0

        # Optimization phase using OOB pixels
        if self.use_adaptive_threshold and hasattr(self.base_estimator_, "oob_decision_function_"):
            if spatial_shape is None or train_idx is None:
                raise ValueError("spatial_shape and train_idx must be provided for OOB optimization.")
            
            t0 = time.time()
            oob_proba = self.base_estimator_.oob_decision_function_
            # Some OOB probabilities might be NaN if a sample was in all trees (rare but possible). Replace with predict_proba.
            nan_mask = np.isnan(oob_proba).any(axis=1)
            if np.any(nan_mask):
                oob_proba[nan_mask] = self.base_estimator_.predict_proba(X[nan_mask])

            # Construct OOB 2D map
            H, W = spatial_shape
            full_proba = np.zeros((H * W, len(self.classes_)), dtype=np.float32)
            full_proba[train_idx] = oob_proba.astype(np.float32)
            prob_map_2d = full_proba.reshape((H, W, len(self.classes_)))

            # Original labels for 2D map (background/test = 0 or invalid, we will use train_idx mask to evaluate)
            y_2d = np.zeros(H * W, dtype=np.int16)
            y_2d[train_idx] = y.astype(np.int16)
            y_2d = y_2d.reshape((H, W))

            # Train mask to evaluate only on training pixels
            train_mask_1d = np.zeros(H * W, dtype=bool)
            train_mask_1d[train_idx] = True
            train_mask = train_mask_1d.reshape((H, W))

            # Apply STEP-B: Diffusion on the OOB map
            if self.use_diffusion:
                for k in range(len(self.classes_)):
                    local_mean = scipy.ndimage.uniform_filter(prob_map_2d[:, :, k], size=3, mode="constant", cval=0.0)
                    prob_map_2d[:, :, k] = 0.7 * prob_map_2d[:, :, k] + 0.3 * local_mean

            # STEP-C: Entropy Map
            # Add 1e-12 to prevent log(0)
            entropy_map = -np.sum(prob_map_2d * np.log(prob_map_2d + 1e-12), axis=2).astype(np.float32)

            # STEP-D: Greedy Class-wise Threshold Optimization
            # Initialize thresholds (start with 90th percentile of class GT entropy)
            for c_idx, c in enumerate(self.classes_):
                c_mask = train_mask & (y_2d == c)
                if np.any(c_mask):
                    self.class_thresholds_[c] = np.percentile(entropy_map[c_mask], 90)
                else:
                    self.class_thresholds_[c] = 999.0 # infinity

            # Candidates: 50, 55, ..., 95
            percentiles_to_test = np.arange(50, 100, 5)

            for c_idx, c in enumerate(self.classes_):
                c_mask = train_mask & (y_2d == c)
                if not np.any(c_mask):
                    continue

                best_score = -1.0
                best_thresh = self.class_thresholds_[c]
                
                # Precalculate percentiles for this class
                candidate_thresholds = np.percentile(entropy_map[c_mask], percentiles_to_test)

                # Base prediction on OOB map
                initial_pred_2d = self.classes_[np.argmax(prob_map_2d, axis=2)].astype(np.int16)

                for thresh in candidate_thresholds:
                    # Temporarily update threshold
                    self.class_thresholds_[c] = thresh
                    
                    # Run STEP-E -> F -> H (Skip G)
                    current_pred = initial_pred_2d.copy()

                    # STEP-E: Entropy Revocation
                    if self.use_entropy_revocation:
                        for cl in self.classes_:
                            th = self.class_thresholds_[cl]
                            revoke_mask = (current_pred == cl) & (entropy_map > th)
                            current_pred[revoke_mask] = -999

                    # STEP-F: Spatial Consistency Revocation
                    if self.use_spatial_revocation:
                        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
                        for cl in self.classes_:
                            mask = (current_pred == cl)
                            neighbor_counts = scipy.ndimage.convolve(mask.astype(np.uint8), kernel, mode="constant", cval=0)
                            revoke_mask = mask & (neighbor_counts <= 3)
                            current_pred[revoke_mask] = -999

                    # STEP-H: Fallback Recovery
                    fallback_mask = (current_pred == -999)
                    current_pred[fallback_mask] = initial_pred_2d[fallback_mask]

                    # Evaluate on training pixels where GT == c (as per request: GT class F1 optimization)
                    # Or evaluate MacroF1 on all training pixels?
                    # The prompt said: "クラス別閾値は predicted class ではなく ground truth class ごとに最適化する"
                    # which means the percentile is chosen from GT class. 
                    # For evaluation metric, we evaluate the overall F1/Kappa on all train pixels to see the global effect.
                    y_true_eval = y_2d[train_mask]
                    y_pred_eval = current_pred[train_mask]

                    if self.optimization_metric == "macro_f1":
                        score = f1_score(y_true_eval, y_pred_eval, average="macro", zero_division=0)
                    else:
                        score = cohen_kappa_score(y_true_eval, y_pred_eval)

                    if score > best_score:
                        best_score = score
                        best_thresh = thresh
                
                # Greedily update to the best threshold found for this class
                self.class_thresholds_[c] = best_thresh

            self.timing_log_["optimization"] = time.time() - t0
        else:
            # Fallback: Fixed 90th percentile
            # We don't have spatial context, so we just use predict_proba on X
            prob_X = self.base_estimator_.predict_proba(X).astype(np.float32)
            entropy_X = -np.sum(prob_X * np.log(prob_X + 1e-12), axis=1)
            for c in self.classes_:
                c_mask = (y == c)
                if np.any(c_mask):
                    self.class_thresholds_[c] = np.percentile(entropy_X[c_mask], 90)
                else:
                    self.class_thresholds_[c] = 999.0
            self.timing_log_["optimization"] = 0.0

        self.timing_log_["fit_total"] = time.time() - start_total
        return self

    def predict(self, X, spatial_shape=None):
        start_total = time.time()
        
        # Reset counters
        self.revoked_pixels_entropy_ = 0
        self.revoked_pixels_spatial_ = 0
        self.reconstructed_pixels_ = 0

        if spatial_shape is None:
            # Fallback to non-spatial prediction if shape not provided
            return self.base_estimator_.predict(X)

        H, W = spatial_shape

        # STEP-A
        t0 = time.time()
        prob_map_1d = self.base_estimator_.predict_proba(X).astype(np.float32)
        prob_map_2d = prob_map_1d.reshape((H, W, len(self.classes_)))
        initial_pred_2d = self.classes_[np.argmax(prob_map_2d, axis=2)].astype(np.int16)
        self.timing_log_["predict_base"] = time.time() - t0

        # STEP-B
        t0 = time.time()
        if self.use_diffusion:
            for k in range(len(self.classes_)):
                local_mean = scipy.ndimage.uniform_filter(prob_map_2d[:, :, k], size=3, mode="constant", cval=0.0)
                prob_map_2d[:, :, k] = 0.7 * prob_map_2d[:, :, k] + 0.3 * local_mean
        self.timing_log_["predict_diffusion"] = time.time() - t0

        # STEP-C
        t0 = time.time()
        entropy_map = -np.sum(prob_map_2d * np.log(prob_map_2d + 1e-12), axis=2).astype(np.float32)
        self.timing_log_["predict_entropy_map"] = time.time() - t0

        current_pred = initial_pred_2d.copy()

        # STEP-E
        t0 = time.time()
        if self.use_entropy_revocation:
            for cl in self.classes_:
                th = self.class_thresholds_.get(cl, 999.0)
                revoke_mask = (current_pred == cl) & (entropy_map > th)
                self.revoked_pixels_entropy_ += np.sum(revoke_mask)
                current_pred[revoke_mask] = -999
        self.timing_log_["predict_entropy_revoc"] = time.time() - t0

        # STEP-F
        t0 = time.time()
        if self.use_spatial_revocation:
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
            for cl in self.classes_:
                mask = (current_pred == cl)
                neighbor_counts = scipy.ndimage.convolve(mask.astype(np.uint8), kernel, mode="constant", cval=0)
                revoke_mask = mask & (neighbor_counts <= 3)
                self.revoked_pixels_spatial_ += np.sum(revoke_mask)
                current_pred[revoke_mask] = -999
        self.timing_log_["predict_spatial_revoc"] = time.time() - t0

        # STEP-G
        t0 = time.time()
        if self.use_reconstruction:
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
            for iteration in range(5):
                unknown_mask = (current_pred == -999)
                remaining_unknown = np.sum(unknown_mask)
                if remaining_unknown == 0:
                    break
                
                # Majority vote for UNKNOWN pixels
                votes = np.zeros((H, W, len(self.classes_)), dtype=np.int16)
                for c_idx, cl in enumerate(self.classes_):
                    cl_mask = (current_pred == cl).astype(np.int16)
                    votes[:, :, c_idx] = scipy.ndimage.convolve(cl_mask, kernel, mode="constant", cval=0)
                
                # Max votes per pixel
                max_votes = np.max(votes, axis=2)
                best_class_idx = np.argmax(votes, axis=2)
                
                # Fill holes where max_votes > 0
                fill_mask = unknown_mask & (max_votes > 0)
                filled_count = np.sum(fill_mask)
                if filled_count == 0:
                    break
                
                current_pred[fill_mask] = self.classes_[best_class_idx[fill_mask]]
                self.reconstructed_pixels_ += filled_count
        self.timing_log_["predict_reconstruction"] = time.time() - t0

        # STEP-H
        t0 = time.time()
        fallback_mask = (current_pred == -999)
        current_pred[fallback_mask] = initial_pred_2d[fallback_mask]
        self.timing_log_["predict_fallback"] = time.time() - t0

        self.timing_log_["predict_total"] = time.time() - start_total
        return current_pred.flatten()

    def predict_proba(self, X):
        return self.base_estimator_.predict_proba(X)
