from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier


class TabNet:
    def __init__(self, n_d=8, n_steps=3, gamma=1.3, lambda_sparse=1e-4, patience=10, max_epochs=100):
        self.model = TabNetClassifier(
            n_d=n_d,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            verbose=1
        )

        self.patience = patience
        self.max_epochs = max_epochs

        self.train_losses = []
        self.val_losses = []

    def fit_eval(self, X_train, y_train, X_val, y_val, batch_size=128, class_weights=None):
        """ Fits the model using native pytorch-tabnet API """

        X_t = X_train.detach().cpu().numpy() if hasattr(X_train, 'cpu') else (X_train.to_numpy() if hasattr(X_train, 'to_numpy') else np.array(X_train))
        y_t = y_train.detach().cpu().numpy() if hasattr(y_train, 'cpu') else (y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.array(y_train))
        X_v = X_val.detach().cpu().numpy() if hasattr(X_val, 'cpu') else (X_val.to_numpy() if hasattr(X_val, 'to_numpy') else np.array(X_val))
        y_v = y_val.detach().cpu().numpy() if hasattr(y_val, 'cpu') else (y_val.to_numpy() if hasattr(y_val, 'to_numpy') else np.array(y_val))

        self.model.fit(
            X_train=X_t,
            y_train=y_t,
            eval_set=[(X_v, y_v)],
            eval_name=['val'],
            eval_metric=['accuracy', 'logloss'],
            max_epochs=self.max_epochs,
            patience=self.patience,
            weights=class_weights,
            batch_size=batch_size,
        )

        self.train_losses = self.model.history['loss']
        self.val_losses = self.model.history['val_logloss']

    def predict_and_evaluate(self, X_test, y_test):
        """ Returns predictions, probabilities, and a dictionary of metrics """

        X_t = X_test.detach().cpu().numpy() if hasattr(X_test, 'cpu') else (X_test.to_numpy() if hasattr(X_test, 'to_numpy') else np.array(X_test))
    
        y_pred = self.model.predict(X_t)
        y_prob = self.model.predict_proba(X_t)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro'),
            "f1": f1_score(y_test, y_pred, average='macro'),
            "roc_auc": roc_auc_score(y_test, y_prob, multi_class='ovr')
        }
        return y_pred, y_prob, metrics
