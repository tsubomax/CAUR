import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

# ============================================================
# Dummy Model (Fallback)
# ============================================================
class DummyFallbackClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name="Unknown", **kwargs):
        self.model_name = model_name
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        print(f"[WARN] {self.model_name} is running in DUMMY mode. Please install required libraries.")
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.classes_[0])


# ============================================================
# TabNet
# ============================================================
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.model_selection import train_test_split

    class AutoEvalTabNetClassifier(TabNetClassifier):
        def fit(self, X, y, **fit_params):
            X = np.array(X, dtype=np.float32, copy=True)
            y = np.array(y, dtype=np.int64, copy=True)
            
            if 'eval_set' not in fit_params:
                unique_classes, counts = np.unique(y, return_counts=True)
                if len(unique_classes) < 2 or np.min(counts) < 2 or len(X) < 10:
                    super().fit(X, y, **fit_params)
                else:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    fit_params['eval_set'] = [(X_val, y_val)]
                    fit_params['patience'] = fit_params.get('patience', 10)
                    super().fit(X_tr, y_tr, **fit_params)
            else:
                super().fit(X, y, **fit_params)

        def predict(self, X):
            X = np.array(X, dtype=np.float32, copy=True)
            return super().predict(X)

except ImportError:
    AutoEvalTabNetClassifier = None

def get_tabnet_classifier(**kwargs):
    if AutoEvalTabNetClassifier is None:
        return DummyFallbackClassifier(model_name="TabNet", **kwargs)
    return AutoEvalTabNetClassifier(verbose=0, device_name="cpu", **kwargs)


# ============================================================
# 1D-CNN (PyTorch + skorch)
# ============================================================
def get_1dcnn_classifier(filters=32, kernel_size=3, learning_rate=0.001, random_state=0):
    try:
        import torch
        from torch import nn
        from skorch import NeuralNetClassifier

        class CNN1DClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self, filters=32, kernel_size=3, learning_rate=0.001,
                         random_state=0, max_epochs=10, batch_size=256):
                self.filters       = filters
                self.kernel_size   = kernel_size
                self.learning_rate = learning_rate
                self.random_state  = random_state
                self.max_epochs    = max_epochs
                self.batch_size    = batch_size
                self._net          = None
                self.classes_      = None

            def fit(self, X, y):
                X = np.array(X, dtype=np.float32, copy=True)
                y = np.array(y, dtype=np.int64, copy=True)
                self.classes_ = np.unique(y)
                num_classes   = len(self.classes_)

                self._label_map    = {lbl: i for i, lbl in enumerate(self.classes_)}
                self._inv_label_map = {i: lbl for lbl, i in self._label_map.items()}
                y_mapped = np.array([self._label_map[lbl] for lbl in y], dtype=np.int64)

                if self.random_state is not None:
                    torch.manual_seed(self.random_state)

                _filters     = self.filters
                _kernel_size = self.kernel_size

                class _FixedModule(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = nn.Conv1d(1, _filters, kernel_size=_kernel_size,
                                               padding=_kernel_size // 2)
                        self.relu  = nn.ReLU()
                        self.pool  = nn.AdaptiveMaxPool1d(1)
                        self.fc    = nn.Linear(_filters, num_classes)

                    def forward(self, X):
                        x = X.unsqueeze(1)
                        x = self.relu(self.conv1(x))
                        x = self.pool(x).squeeze(2)
                        return self.fc(x)

                self._net = NeuralNetClassifier(
                    module=_FixedModule,
                    criterion=nn.CrossEntropyLoss,
                    optimizer=torch.optim.Adam,
                    lr=self.learning_rate,
                    max_epochs=self.max_epochs,
                    batch_size=self.batch_size,
                    train_split=None,
                    verbose=0,
                    device="cpu",
                )
                self._net.fit(X, y_mapped)
                return self

            def predict(self, X):
                X = np.array(X, dtype=np.float32, copy=True)
                y_mapped = self._net.predict(X)
                return np.array([self._inv_label_map[i] for i in y_mapped])

            def predict_proba(self, X):
                X = np.array(X, dtype=np.float32, copy=True)
                return self._net.predict_proba(X)

        return CNN1DClassifier(
            filters=filters,
            kernel_size=kernel_size,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    except ImportError:
        return DummyFallbackClassifier(
            model_name="1D-CNN", filters=filters, kernel_size=kernel_size, learning_rate=learning_rate
        )

# ============================================================
# EPF Wrapper
# ============================================================
class NearestNeighborFeatureAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        from sklearn.neighbors import NearestNeighbors
        self.nn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            n_jobs=self.n_jobs
        )
        self.nn_.fit(X)
        self.X_train_ = X
        return self

    def transform(self, X):
        _, idx = self.nn_.kneighbors(X)
        neighbor_features = np.mean(self.X_train_[idx[:, 1:]], axis=1)
        return np.hstack([X, neighbor_features])

# ============================================================
# Probabilistic LinearSVC
# ============================================================
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

class ProbabilisticLinearSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        counts = np.bincount(y) if np.issubdtype(type(y[0]), np.integer) else np.unique(y, return_counts=True)[1]
        min_class = np.min(counts[counts > 0])
        cv_folds = min(5, min_class)
        if cv_folds < 2:
            base_lsvc = LinearSVC(random_state=self.random_state)
            base_lsvc.fit(X, y)
            self.model = CalibratedClassifierCV(base_lsvc, cv='prefit')
            self.model.fit(X, y)
        else:
            self.model = CalibratedClassifierCV(LinearSVC(random_state=self.random_state), cv=cv_folds)
            self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

