import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv1D, Dense, Dropout, LayerNormalization, Input, ReLU,
    Concatenate, MaxPooling1D, Reshape, BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
FIXED_THRESHOLD = 0.5
MAX_SEQ_LENGTH = 50

class ECA(Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super(ECA, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = Conv1D(1, kernel_size=self.kernel_size, padding='same', use_bias=False)
    
    def call(self, inputs, training=None):
        # inputs: (batch, features)
        x = tf.expand_dims(inputs, axis=-1)  # (batch, features, 1)
        attn = self.conv(x)  # (batch, features, 1)
        attn = tf.squeeze(attn, axis=-1)  # (batch, features)
        attn = tf.nn.sigmoid(attn)
        return inputs * attn
    
    def get_config(self):
        config = super(ECA, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

class ApplySequenceMask(Layer):

    def call(self, inputs, training=None):
        features, mask = inputs
        mask = tf.cast(mask, features.dtype)
        return features * tf.expand_dims(mask, axis=-1)

class MaskedGlobalPooling1D(Layer):

    def call(self, inputs, training=None):
        features, mask = inputs
        mask = tf.cast(mask, features.dtype)
        mask_3d = tf.expand_dims(mask, axis=-1)

        feature_sum = tf.reduce_sum(features * mask_3d, axis=1)
        valid_count = tf.maximum(tf.reduce_sum(mask_3d, axis=1), 1.0)
        average = feature_sum / valid_count

        negative_large = tf.cast(-1e9, features.dtype)
        masked_features = tf.where(
            tf.cast(mask_3d, tf.bool),
            features,
            negative_large
        )
        maximum = tf.reduce_max(masked_features, axis=1)
        return tf.concat([average, maximum], axis=-1)

class GatedBilinearFusion(Layer):
    def __init__(self, output_dim=128, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # projection
        self.proj1 = Dense(self.output_dim, kernel_regularizer=l2(1e-4))
        self.proj2 = Dense(self.output_dim, kernel_regularizer=l2(1e-4))

        # bilinear interaction
        self.bilinear_dense = Dense(self.output_dim, kernel_regularizer=l2(1e-4))

        # concat branch
        self.concat_dense = Dense(self.output_dim, kernel_regularizer=l2(1e-4))

        # gate（关键）
        self.gate_dense = Dense(self.output_dim, activation='sigmoid')

        # regularization
        self.dropout = Dropout(self.dropout_rate)
        self.norm = LayerNormalization()

    def call(self, inputs, training=None):
        f1, f2 = inputs

        f1_proj = self.proj1(f1)
        f2_proj = self.proj2(f2)

        interaction = f1_proj * f2_proj
        interaction = tf.nn.swish(interaction)
        bilinear_out = self.bilinear_dense(interaction)
        bilinear_out = self.dropout(bilinear_out, training=training)

        concat = tf.concat([f1_proj, f2_proj], axis=-1)
        concat_out = self.concat_dense(concat)
        concat_out = self.dropout(concat_out, training=training)

        gate_input = tf.concat([f1_proj, f2_proj, interaction], axis=-1)
        gate = self.gate_dense(gate_input)

        output = gate * bilinear_out + (1 - gate) * concat_out

        output = self.norm(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

def build_model(max_seq_length=50):

    inp_t5 = Input(shape=(max_seq_length, 1024), name="prot_t5_residue_input")
    inp_t5_mask = Input(shape=(max_seq_length,), name="prot_t5_residue_mask")

    branch3 = Conv1D(64, kernel_size=3, padding="same", use_bias=False,
                     kernel_regularizer=l2(2e-4))(inp_t5)
    branch3 = LayerNormalization(axis=-1)(branch3)
    branch3 = ReLU()(branch3)
    branch3 = ApplySequenceMask()([branch3, inp_t5_mask])

    branch5 = Conv1D(64, kernel_size=5, padding="same", use_bias=False,
                     kernel_regularizer=l2(2e-4))(inp_t5)
    branch5 = LayerNormalization(axis=-1)(branch5)
    branch5 = ReLU()(branch5)
    branch5 = ApplySequenceMask()([branch5, inp_t5_mask])

    branch7 = Conv1D(64, kernel_size=7, padding="same", use_bias=False,
                     kernel_regularizer=l2(2e-4))(inp_t5)
    branch7 = LayerNormalization(axis=-1)(branch7)
    branch7 = ReLU()(branch7)
    branch7 = ApplySequenceMask()([branch7, inp_t5_mask])

    x = Concatenate()([branch3, branch5, branch7])  # (max_seq_length, 192)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=5, padding="same", use_bias=False,
               kernel_regularizer=l2(2e-4))(x)
    x = LayerNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = ApplySequenceMask()([x, inp_t5_mask])
    x = MaxPooling1D(pool_size=2)(x)  # 降低序列维度
    mask_3d = Reshape((max_seq_length, 1))(inp_t5_mask)
    pooled_mask_3d = MaxPooling1D(pool_size=2)(mask_3d)
    pooled_mask = Reshape((max_seq_length // 2,))(pooled_mask_3d)
    x = ApplySequenceMask()([x, pooled_mask])
    x = Dropout(0.3)(x)

    x_pooled = MaskedGlobalPooling1D()([x, pooled_mask])  # (256,)

    x = Dense(256, activation="relu", kernel_regularizer=l2(2e-4))(x_pooled)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    inp_esm = Input(shape=(1280,), name="esm_input")

    y = Dense(512, activation="relu", kernel_regularizer=l2(2e-4))(inp_esm)
    y = BatchNormalization()(y)

    y = ECA(kernel_size=5)(y)
    
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y)

    y = Dense(256, activation="relu", kernel_regularizer=l2(2e-4))(y)
    y = Dropout(0.3)(y)

    fused = GatedBilinearFusion(output_dim=256, dropout_rate=0.3)([x, y])
    fused = BatchNormalization()(fused)
    fused = Dropout(0.4)(fused)

    fused = Dense(128, activation="relu", kernel_regularizer=l2(2e-4))(fused)
    fused = Dropout(0.4)(fused)

    out = Dense(1, activation="sigmoid")(fused)

    model = Model(inputs=[inp_t5, inp_esm, inp_t5_mask], outputs=out)
    return model


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_NAME, help="")
    parser.add_argument("--scaler", default=SCALER_NAME, help="")
    parser.add_argument("--labels", default="")
    parser.add_argument("--prott5", default="")
    parser.add_argument("--esm", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    return args


def load_kelm_data(label_path, residue_path, esm_path):
    metadata = pd.read_csv(label_path, dtype=str, keep_default_na=False)
    required = {"sequence", "label"}

    with residue_path.open("rb") as handle:
        residues = pickle.load(handle)

    padded = np.zeros((len(metadata), MAX_SEQ_LENGTH, 1024), dtype=np.float32)
    mask = np.zeros((len(metadata), MAX_SEQ_LENGTH), dtype=np.float32)
    for index, row in metadata.iterrows():
        seq_id, sequence = row["seq_id"], row["sequence"]
        length = len(sequence)
        padded[index, :length] = embedding
        mask[index, :length] = 1.0

    esm = pd.read_csv(esm_path, header=None).to_numpy(dtype=np.float32)
    return metadata, padded, mask, esm


def evaluate(labels, probabilities):
    probabilities = np.asarray(probabilities).reshape(-1)
    predicted = (probabilities >= FIXED_THRESHOLD).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    return {
        "dataset": "kelm", "n_samples": len(labels),
        "threshold": FIXED_THRESHOLD,
        "acc": accuracy_score(labels, predicted), "bacc": (sn + sp) / 2,
        "sn": sn, "sp": sp, "mcc": matthews_corrcoef(labels, predicted),
        "auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    args = parse_args()
    paths = {name: resolve_path(getattr(args, name))
             for name in ("model", "scaler", "labels", "prott5", "esm")}
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]

    gpus = tf.config.list_physical_devices("GPU")
    metadata, prott5, mask, esm_raw = load_kelm_data(
        paths["labels"], paths["prott5"], paths["esm"])
    with paths["scaler"].open("rb") as handle:
        scaler = pickle.load(handle)
    esm = np.asarray(scaler.transform(esm_raw), dtype=np.float32)

    model = build_model(max_seq_length=MAX_SEQ_LENGTH)
    model.load_weights(str(paths["model"]))
    probabilities = model.predict([prott5, esm, mask], batch_size=args.batch_size,
                                  verbose=1).reshape(-1)
    metrics = evaluate(metadata["label"].to_numpy(), probabilities)
    print("\nKELM Test Results")
    for name in ("acc", "bacc", "sn", "sp", "mcc", "auc", "pr_auc"):
        print(f"{name.upper():7s}: {metrics[name]:.4f}")
    print(f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")

    output_root = resolve_path(args.output_dir)
    output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = metadata[["seq_id", "sequence", "label"]].rename(columns={"label": "y_true"}).copy()
    predictions["sequence_length"] = predictions["sequence"].str.len()
    predictions["y_pred_proba"] = probabilities
    predictions["y_pred"] = (probabilities >= FIXED_THRESHOLD).astype(np.int32)


if __name__ == "__main__":
    main()
