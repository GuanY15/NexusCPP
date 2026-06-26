"""
model_A1 + ECA + GBF 单次训练
=========================================
- ProtT5通道：多尺度Conv(3,5,7)
- ESM通道：ECA (Efficient Channel Attention) - 轻量级通道注意力，无降维
- 融合：GBF (Gated Bilinear Fusion)

与 model_A1_ECA_GBF_100runs.py 完全一致的模型结构和训练参数
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import * 
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score, roc_auc_score, confusion_matrix
import os
from datetime import datetime

# ============================
#  配置
# ============================   

DATA_SPLIT_SEED = 333
RESULTS_DIR = "train_results"


# ============================
#  ECA (Efficient Channel Attention)
# ============================

class ECA(Layer):
    """
    Efficient Channel Attention
    使用1D卷积实现局部跨通道交互，无需降维
    优点：参数少，避免降维信息损失
    """
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

        # gate（关键）
        self.gate_dense = Dense(self.output_dim, activation='sigmoid')

        # regularization
        self.dropout = Dropout(self.dropout_rate)
        self.norm = LayerNormalization()

    def call(self, inputs, training=None):
        f1, f2 = inputs

        # 1️⃣ projection
        f1_proj = self.proj1(f1)
        f2_proj = self.proj2(f2)

        # 2️⃣ bilinear interaction
        interaction = f1_proj * f2_proj
        interaction = tf.nn.swish(interaction)   # 标准推荐
        bilinear_out = self.bilinear_dense(interaction)
        bilinear_out = self.dropout(bilinear_out, training=training)

        # 3️⃣ concat branch
        concat = tf.concat([f1_proj, f2_proj], axis=-1)
        concat_out = self.concat_dense(concat)
        concat_out = self.dropout(concat_out, training=training)

        # 4️⃣ gate（标准写法：基于交互信息）
        gate_input = tf.concat([f1_proj, f2_proj, interaction], axis=-1)
        gate = self.gate_dense(gate_input)

        # 5️⃣ fusion
        output = gate * bilinear_out + (1 - gate) * concat_out

        # 6️⃣ normalization（稳定训练）
        output = self.norm(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


# ============================
#  模型结构 (多尺度Conv + ECA + GBF)
# ============================

def build_model():
    """
    - ProtT5通道：多尺度Conv(3,5,7)
    - ESM通道：ECA (替换SE Block)
    - 融合：GBF
    """
    # -------- ProtT5 通道：多尺度Conv --------
    inp_t5 = Input(shape=(1024, 1), name="prot_t5_input")

    branch3 = Conv1D(32, 3, padding="same", kernel_regularizer=l2(1e-4))(inp_t5)
    branch5 = Conv1D(32, 5, padding="same", kernel_regularizer=l2(1e-4))(inp_t5)
    branch7 = Conv1D(32, 7, padding="same", kernel_regularizer=l2(1e-4))(inp_t5)
    
    x = Concatenate()([branch3, branch5, branch7])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)

    # -------- ESM 通道：ECA (Efficient Channel Attention) --------
    inp_esm = Input(shape=(1280,), name="esm_input")
    
    # 基础映射
    y = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(inp_esm)
    y = BatchNormalization()(y)
    
    # ECA: 轻量级通道注意力（无降维）
    y = ECA(kernel_size=5)(y)
    
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y)
    
    # 降维到256
    y = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(y)
    y = Dropout(0.3)(y)

    # -------- 融合：GBF --------
    fused = GatedBilinearFusion(output_dim=256)([x, y])
    fused = BatchNormalization()(fused)
    fused = Dropout(0.4)(fused)
    
    # 分类器
    fused = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(fused)
    fused = Dropout(0.4)(fused)

    out = Dense(1, activation="sigmoid")(fused)

    model = Model(inputs=[inp_t5, inp_esm], outputs=out)
    return model


# ============================
#  数据加载
# ============================

print("Loading data...")

prot_t5_path = r"E:\PythonProject\pLM4CPPs\featureextract\last\prot_t5_xl_uniref50_mean.csv"
esm_path = r"E:\PythonProject\pLM4CPPs\featureextract\last\esm2_t33_650M_UR50D_mean.csv"
label_path = r"E:\PythonProject\pLM4CPPs\embedded_data\last.csv"

prot_t5_raw = pd.read_csv(prot_t5_path, header=None).values
esm_raw = pd.read_csv(esm_path, header=None).values
labels = pd.read_csv(label_path)["label"].values

kelm_label_path = r"E:\PythonProject\pLM4CPPs\embedded_data\kelm_dataset_CPP.csv"
kelm_prot_t5_path = r"E:\PythonProject\pLM4CPPs\featureextract\kelm\prot_t5_xl_uniref50_mean.csv"
kelm_esm_path = r"E:\PythonProject\pLM4CPPs\featureextract\kelm\esm2_t33_650M_UR50D_mean.csv"

kelm_labels = pd.read_csv(kelm_label_path, na_filter=False)["label"].values
kelm_prot_t5_raw = pd.read_csv(kelm_prot_t5_path, header=None).values
kelm_esm_raw = pd.read_csv(kelm_esm_path, header=None).values

print(f"Training data: {prot_t5_raw.shape[0]} samples")
print(f"KELM data: {kelm_prot_t5_raw.shape[0]} samples")


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


def evaluate(y_true, y_pred_proba):
    best_threshold, _ = optimize_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba > best_threshold).astype(int).flatten()
    
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    bacc = (sensitivity + specificity) / 2
    
    return {'acc': acc, 'bacc': bacc, 'mcc': mcc, 'auc': auc,
            'sn': sensitivity, 'sp': specificity, 'threshold': best_threshold}


# ============================
#  主程序
# ============================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("model_train")
    print(f"Data split seed: {DATA_SPLIT_SEED}")
    print("="*60)
    
    start_time = datetime.now()

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

    X2_train = scaler2.fit_transform(X2_train_raw)
    X2_test = scaler2.transform(X2_test_raw)

    # ProtT5 reshape
    X1_train = X1_train.reshape(len(X1_train), 1024, 1)
    X1_test = X1_test.reshape(len(X1_test), 1024, 1)
    
    # KELM 数据
    kelm_prot_t5 = scaler1.transform(kelm_prot_t5_raw)
    kelm_esm = scaler2.transform(kelm_esm_raw)
    kelm_prot_t5 = kelm_prot_t5.reshape(len(kelm_prot_t5), 1024, 1)
    
    print(f"Train: {X1_train.shape[0]} | Test: {X1_test.shape[0]} | KELM: {kelm_prot_t5.shape[0]}")

    # 构建模型
    tf.keras.backend.clear_session()
    
    model = build_model()
    model.summary()

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  metrics=["accuracy"])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=20,
            restore_best_weights=True, verbose=1)
    ]
    
    # 训练
    print("\nTraining...")
    history = model.fit([X1_train, X2_train], y_train,
              validation_data=([X1_test, X2_test], y_test),
              epochs=100, batch_size=32, class_weight={0: 1, 1: 2},
              callbacks=callbacks, verbose=1)
    
    # 评估 Internal Test
    print("\n" + "="*60)
    print("Internal Test Results")
    print("="*60)
    internal_pred = model.predict([X1_test, X2_test], verbose=0)
    internal_result = evaluate(y_test, internal_pred)
    
    print(f"ACC:  {internal_result['acc']:.4f}")
    print(f"BACC: {internal_result['bacc']:.4f}")
    print(f"Sn:   {internal_result['sn']:.4f}")
    print(f"Sp:   {internal_result['sp']:.4f}")
    print(f"MCC:  {internal_result['mcc']:.4f}")
    print(f"AUC:  {internal_result['auc']:.4f}")
    print(f"Threshold: {internal_result['threshold']:.2f}")
    
    # 评估 KELM
    print("\n" + "="*60)
    print("KELM Test Results")
    print("="*60)
    kelm_pred = model.predict([kelm_prot_t5, kelm_esm], verbose=0)
    kelm_result = evaluate(kelm_labels, kelm_pred)
    
    print(f"ACC:  {kelm_result['acc']:.4f}")
    print(f"BACC: {kelm_result['bacc']:.4f}")
    print(f"Sn:   {kelm_result['sn']:.4f}")
    print(f"Sp:   {kelm_result['sp']:.4f}")
    print(f"MCC:  {kelm_result['mcc']:.4f}")
    print(f"AUC:  {kelm_result['auc']:.4f}")
    print(f"Threshold: {kelm_result['threshold']:.2f}")

    # 保存模型
    model_path = os.path.join(RESULTS_DIR, "trained_model.h5")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    print(f"Total time: {datetime.now() - start_time}")
    print("Done!")
