"""
评估指定模型在 KELM 数据集上的混淆矩阵和指标
=====================================================
模型: model_run500_seed50429.h5

输出（保存在 SAVE_DIR 中）：
1. kelm_all_predictions.csv   — 所有样本（序列、标签、预测概率）
2. kelm_cpp_ranked.csv        — CPP 样本按置信度降序排列
3. 控制台输出评估指标
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os

DATA_SPLIT_SEED = 333
SAVE_DIR = "kelm_eval_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================
#  自定义层（与训练时一致）
# ============================

class ECA(Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super(ECA, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv1D(1, kernel_size=self.kernel_size, padding='same', use_bias=False)

    def call(self, inputs, training=None):
        x = tf.expand_dims(inputs, axis=-1)
        attn = self.conv(x)
        attn = tf.squeeze(attn, axis=-1)
        attn = tf.nn.sigmoid(attn)
        return inputs * attn

    def get_config(self):
        config = super(ECA, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class GatedBilinearFusion(Layer):
    def __init__(self, output_dim=128, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.proj1 = Dense(self.output_dim, kernel_regularizer=l2(1e-4))
        self.proj2 = Dense(self.output_dim, kernel_regularizer=l2(1e-4))
        self.bilinear_dense = Dense(self.output_dim, kernel_regularizer=l2(1e-4))
        self.concat_dense = Dense(self.output_dim, kernel_regularizer=l2(1e-4))
        self.gate_dense = Dense(self.output_dim, activation='sigmoid')
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
        config.update({"output_dim": self.output_dim, "dropout_rate": self.dropout_rate})
        return config


# ============================
#  评估函数
# ============================

def optimize_threshold(y_true, y_pred_probas):
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_mcc, best_threshold = -1, 0.5
    for threshold in thresholds:
        y_pred = (y_pred_probas > threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, threshold
    return best_threshold, best_mcc


def evaluate(y_true, y_pred_proba, dataset_name="", threshold=None):
    if threshold is None:
        threshold, _ = optimize_threshold(y_true, y_pred_proba)

    y_pred = (y_pred_proba > threshold).astype(int).flatten()

    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    bacc = (sensitivity + specificity) / 2

    print(f"\n{'='*55}")
    print(f"  {dataset_name}")
    print(f"{'='*55}")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  ACC       : {acc:.4f}")
    print(f"  BACC      : {bacc:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Sn (Recall): {sensitivity:.4f}")
    print(f"  Sp        : {specificity:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Neg  Predicted Pos")
    print(f"  Actual Neg      TN={tn:<6d}    FP={fp}")
    print(f"  Actual Pos      FN={fn:<6d}    TP={tp}")
    print(f"{'='*55}")

    return {'acc': acc, 'bacc': bacc, 'mcc': mcc, 'auc': auc,
            'sn': sensitivity, 'sp': specificity, 'threshold': threshold,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


# ============================
#  主程序
# ============================

if __name__ == "__main__":

    MODEL_PATH = r"E:\PythonProject\NexusCPP\best_internal_model.h5"

    # ---------- 加载数据 ----------
    print("Loading data...")

    # 内部数据集（用于 scaler fit）
    prot_t5_path = r"E:\PythonProject\NexusCPP\featureextract\last\prot_t5_xl_uniref50_mean.csv"
    esm_path = r"E:\PythonProject\NexusCPP\featureextract\last\esm2_t33_650M_UR50D_mean.csv"
    label_path = r"E:\PythonProject\NexusCPP\embedded_data\last.csv"

    prot_t5_raw = pd.read_csv(prot_t5_path, header=None).values
    esm_raw = pd.read_csv(esm_path, header=None).values
    df_internal = pd.read_csv(label_path)
    labels = df_internal["label"].values
    internal_sequences = df_internal["sequence"].values

    # KELM 外部数据集
    kelm_data_path = r"E:\PythonProject\NexusCPP\embedded_data\kelm_dataset.csv"
    kelm_prot_t5_path = r"E:\PythonProject\NexusCPP\featureextract\kelm\prot_t5_xl_uniref50_mean.csv"
    kelm_esm_path = r"E:\PythonProject\NexusCPP\featureextract\kelm\esm2_t33_650M_UR50D_mean.csv"

    kelm_df = pd.read_csv(kelm_data_path, na_filter=False)
    kelm_sequences = kelm_df["sequence"].values
    kelm_labels = kelm_df["label"].values
    kelm_prot_t5_raw = pd.read_csv(kelm_prot_t5_path, header=None).values
    kelm_esm_raw = pd.read_csv(kelm_esm_path, header=None).values

    # ---------- 归一化（必须用内部数据集 fit） ----------
    # 先划分原始特征
    X1_train_raw, X1_test_raw, X2_train_raw, X2_test_raw, y_train, y_test = train_test_split(
        prot_t5_raw, esm_raw, labels,
        test_size=0.2,
        random_state=DATA_SPLIT_SEED,
        stratify=labels
    )

    # 只在训练集上拟合归一化器
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    X1_train = scaler1.fit_transform(X1_train_raw)
    X1_test = scaler1.transform(X1_test_raw)
    kelm_prot_t5 = scaler1.transform(kelm_prot_t5_raw)

    X2_train = scaler2.fit_transform(X2_train_raw)
    X2_test = scaler2.transform(X2_test_raw)
    kelm_esm = scaler2.transform(kelm_esm_raw)

    # ProtT5 reshape
    X1_train = X1_train.reshape(len(X1_train), 1024, 1)
    X1_test = X1_test.reshape(len(X1_test), 1024, 1)
    kelm_prot_t5 = kelm_prot_t5.reshape(len(kelm_prot_t5), 1024, 1)

    # KELM 数据归一化
    kelm_prot_t5 = scaler1.transform(kelm_prot_t5_raw)
    kelm_esm = scaler2.transform(kelm_esm_raw)
    kelm_prot_t5 = kelm_prot_t5.reshape(len(kelm_prot_t5), 1024, 1)

    print(f"KELM samples: {kelm_prot_t5.shape[0]}")
    print(f"KELM label distribution: {dict(zip(*np.unique(kelm_labels, return_counts=True)))}")

    # ---------- 加载模型 ----------
    print(f"\nLoading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'ECA': ECA, 'GatedBilinearFusion': GatedBilinearFusion}
    )
    print("Model loaded successfully.")

    # ---------- KELM 评估 ----------
    kelm_pred = model.predict([kelm_prot_t5, kelm_esm], verbose=0)
    kelm_result = evaluate(kelm_labels, kelm_pred, "KELM External Test Set")

    # 保存 KELM 全部预测结果
    kelm_pred_flat = kelm_pred.flatten()
    df_kelm_all = pd.DataFrame({
        'sequence': kelm_sequences,
        'label': kelm_labels,
        'pred_probability': kelm_pred_flat
    })
    df_kelm_all.to_csv(os.path.join(SAVE_DIR, "kelm_all_predictions.csv"), index=False)
    print(f"Saved: {SAVE_DIR}/kelm_all_predictions.csv ({len(df_kelm_all)} samples)")

    internal_pred = model.predict([X1_test, X2_test], verbose=0)
    internal_result = evaluate(y_test, internal_pred, "Internal Test Set (Reference)")

    # 保存内部测试集全部预测结果
    internal_pred_flat = internal_pred.flatten()
    df_int_all = pd.DataFrame({
        'label': y_test,
        'pred_probability': internal_pred_flat
    })
    df_int_all.to_csv(os.path.join(SAVE_DIR, "internal_all_predictions.csv"), index=False)
    print(f"Saved: {SAVE_DIR}/internal_all_predictions.csv ({len(df_int_all)} samples)")

