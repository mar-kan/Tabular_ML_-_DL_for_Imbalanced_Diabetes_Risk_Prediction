from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabtransformertf.models.fttransformer import FTTransformer, FTTransformerEncoder
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np


class FTTransformerWrapper:
    def __init__(
        self,
        learning_rate=1e-3,
        depth=3,
        heads=4,
        attn_dropout=0.1,
        ff_dropout=0.1,
        embedding_dim=32,
        batch_size=256,
        patience=3
    ):
        self.lr = learning_rate
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.patience = patience

        self.model = None
        self.history = None

    def _create_feature_dict(self, X, num_cols, cat_cols):
        features = {}

        for col in num_cols:
            features[col] = X[[col]].to_numpy(dtype=np.float32)

        for col in cat_cols:
            values = X[[col]].astype(str).fillna("NA").to_numpy(dtype=str)
            features[col] = tf.convert_to_tensor(values, dtype=tf.string)

        return features

    def build_model(self, num_cols, cat_cols, X_train):

        cat_data = X_train[cat_cols].astype(str).fillna("NA").to_numpy(dtype=str)
        num_data = X_train[num_cols].to_numpy(dtype=np.float32)

        encoder = FTTransformerEncoder(
            numerical_features=num_cols,
            categorical_features=cat_cols,
            numerical_data=num_data,
            categorical_data=cat_data,
            depth=self.depth,
            heads=self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            embedding_dim=self.embedding_dim
        )

        self.model = FTTransformer(
            numerical_features=num_cols,
            categorical_features=cat_cols,
            out_activation="softmax",
            out_dim=3,
            encoder=encoder
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def fit(self, X_train, y_train, X_val, y_val, num_cols, cat_cols, epochs=100):
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )

        if self.model is None:
            self.build_model(num_cols, cat_cols, X_train)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

        x_train = self._create_feature_dict(X_train, num_cols, cat_cols)
        x_val = self._create_feature_dict(X_val, num_cols, cat_cols)

        self.history = self.model.fit(
            x=x_train,
            y=y_train.astype(np.int32),
            validation_data=(x_val, y_val.astype(np.int32)),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            sample_weight=sample_weights,
            verbose=1
        )

    def predict_and_evaluate(self, X_test, y_test, num_cols, cat_cols):

        x_test = self._create_feature_dict(X_test, num_cols, cat_cols)
        probs = self.model.predict(x_test, verbose=0)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro')
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')
        roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')

        return preds, probs, {
            'accuracy': acc, 
            'precision': prec, 
            'recall': rec, 
            'f1': f1, 
            'roc_auc': roc_auc
        }
