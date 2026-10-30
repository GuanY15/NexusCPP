import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

DATA_SPLIT_SEED = 333
FIXED_THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LENGTH = 50

# ============================
#  ECA (Efficient Channel Attention)
# ============================

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


# ============================
#  Gated Bilinear Fusion (GBF)
# ============================

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

        # gate
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

def load_residue_embeddings(pkl_path):
    return data

def pad_residue_embeddings(embeddings_dict, label_df, max_length=MAX_SEQ_LENGTH):

    n_samples = len(label_df)
    padded_embeddings = np.zeros(
        (n_samples, max_length, 1024), dtype=np.float32
    )
    sequence_mask = np.zeros((n_samples, max_length), dtype=np.float32)
    sequence_lengths = np.zeros(n_samples, dtype=np.int32)
    labels = label_df['label'].to_numpy(dtype=np.int32)

    for row_index, row in label_df.iterrows():
        seq_id = str(row['seq_id'])
        sequence = str(row['sequence'])
        emb = np.asarray(embeddings_dict[seq_id], dtype=np.float32)
        seq_len = len(sequence)

        padded_embeddings[row_index, :seq_len] = emb
        sequence_mask[row_index, :seq_len] = 1.0
        sequence_lengths[row_index] = seq_len

    return padded_embeddings, sequence_mask, sequence_lengths, labels

label_path = PROJECT_ROOT

label_df = pd.read_csv(label_path, dtype={'id': str})

label_df['seq_id'] = label_df['id'].astype(str)

prot_t5_residue_path = PROJECT_ROOT

print("\nLoading residue-level embeddings...")
prot_t5_residue_dict = load_residue_embeddings(prot_t5_residue_path)

print("\nPadding residue embeddings...")
prot_t5_padded, prot_t5_mask, sequence_lengths, labels = pad_residue_embeddings(
    prot_t5_residue_dict, label_df, MAX_SEQ_LENGTH
)
esm_path = PROJECT_ROOT
esm_raw = pd.read_csv(esm_path, header=None).to_numpy(dtype=np.float32)


def evaluate(y_true, y_pred_proba, threshold=FIXED_THRESHOLD):

    y_pred_proba = np.asarray(y_pred_proba).reshape(-1)
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    bacc = (sensitivity + specificity) / 2

    return {'acc': acc, 'bacc': bacc, 'mcc': mcc, 'auc': auc, 'pr_auc': pr_auc,
            'sn': sensitivity, 'sp': specificity, 'threshold': threshold}


def build_prediction_table(metadata_df, indices, probabilities, threshold):
    subset = metadata_df.iloc[np.asarray(indices)].reset_index(drop=True)
    probabilities = np.asarray(probabilities).reshape(-1)

    result = subset[['seq_id', 'sequence', 'label']].copy()
    result = result.rename(columns={'label': 'y_true'})
    result['sequence_length'] = result['sequence'].str.len()
    result['y_pred_proba'] = probabilities
    result['y_pred'] = (probabilities >= threshold).astype(np.int32)
    return result

if __name__ == "__main__":

    start_time = datetime.now()
    all_indices = np.arange(len(labels))
    trainval_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.1,
        random_state=DATA_SPLIT_SEED,
        stratify=labels
    )

    train_indices, val_indices = train_test_split(
        trainval_indices,
        test_size=0.1111,
        random_state=DATA_SPLIT_SEED,
        stratify=labels[trainval_indices]
    )

    X1_train = prot_t5_padded[train_indices]
    X1_val = prot_t5_padded[val_indices]
    X1_test = prot_t5_padded[test_indices]
    M1_train = prot_t5_mask[train_indices]
    M1_val = prot_t5_mask[val_indices]
    M1_test = prot_t5_mask[test_indices]
    X2_train_raw = esm_raw[train_indices]
    X2_val_raw = esm_raw[val_indices]
    X2_test_raw = esm_raw[test_indices]
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    classes = np.array([0, 1], dtype=np.int32)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    print(f"Training class weights: {class_weights}")

    del prot_t5_padded, prot_t5_mask, esm_raw, sequence_lengths

    scaler2 = MinMaxScaler()
    X2_train = scaler2.fit_transform(X2_train_raw).astype(np.float32)
    X2_val = scaler2.transform(X2_val_raw).astype(np.float32)
    X2_test = scaler2.transform(X2_test_raw).astype(np.float32)


    print("\n" + "="*60)
    print("Building model...")
    print("="*60)

    model = build_model(max_seq_length=MAX_SEQ_LENGTH)
    model.summary()

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                  metrics=[tf.keras.metrics.BinaryAccuracy(
                      name="accuracy", threshold=FIXED_THRESHOLD)])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=20,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8,
            min_lr=1e-5, verbose=1)
    ]

    print("\n" + "="*60)
    print("Training...")
    print("="*60)

    val_pred = model.predict([X1_val, X2_val, M1_val], verbose=0)

    val_result = evaluate(y_val, val_pred, threshold=FIXED_THRESHOLD)
    print(f"\nValidation Set Performance (at threshold={FIXED_THRESHOLD:.2f}):")
    print(f"  ACC:  {val_result['acc']:.4f}")
    print(f"  BACC: {val_result['bacc']:.4f}")
    print(f"  Sn:   {val_result['sn']:.4f}")
    print(f"  Sp:   {val_result['sp']:.4f}")
    print(f"  MCC:  {val_result['mcc']:.4f}")
    print(f"  AUC:  {val_result['auc']:.4f}")
    print(f"  PR-AUC: {val_result['pr_auc']:.4f}")

    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)

    test_pred = model.predict([X1_test, X2_test, M1_test], verbose=0)
    test_result = evaluate(y_test, test_pred, threshold=FIXED_THRESHOLD)

    print(f"ACC:  {test_result['acc']:.4f}")
    print(f"BACC: {test_result['bacc']:.4f}")
    print(f"Sn:   {test_result['sn']:.4f}")
    print(f"Sp:   {test_result['sp']:.4f}")
    print(f"MCC:  {test_result['mcc']:.4f}")
    print(f"AUC:  {test_result['auc']:.4f}")
    print(f"PR-AUC: {test_result['pr_auc']:.4f}")
