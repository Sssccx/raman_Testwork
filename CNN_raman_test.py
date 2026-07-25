import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 全局配置
BASE_DIR = "D:/python_homework/Raman_spectral_signature_analysis/Raman_Data"  # 解压后所有文件夹所在的根目录
EXCEL_PATH = "D:/python_homework/Raman_spectral_signature_analysis/Raman_Data/clinical characters of patients.xls"  # 病理信息表
COMMON_AXIS = np.linspace(400, 1800, 700)  # 统一拉曼位移轴
SG_WINDOW = 9  # 平滑窗口
SG_POLY = 3  # 多项式阶数
N_SPLITS = 5  # 交叉验证折数
BATCH_SIZE = 32
EPOCHS = 250
PATIENCE = 30
RANDOM_STATE = 42
VAL_PATIENT_RATIO = 0.25   # 训练患者中划分给验证集的比例
AUGMENT_TRAIN = 1


# 从文件名提取患者 ID
def extract_patient_id(filename: str) -> str:

    stem = Path(filename).stem
    base = re.sub(r'-\d+$', '', stem)
    parts = base.split('_')
    patient_id = parts[-1]
    return patient_id



# 读取临床信息表
def load_clinical_info(excel_path):
    """读取 clinical characters of patients.xls，返回字典：{患者ID: 标签(0/1)}"""
    df = pd.read_excel(excel_path, engine='xlrd')
    id_col = '编号'
    label_col = '是否骨转移（1,是。0，否）'
    label_map = {'with BM': 1, 'without BM': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0}
    clinical_dict = {}
    for _, row in df.iterrows():
        pid = str(row[id_col]).strip()
        lab = row[label_col]
        if isinstance(lab, str):
            lab = label_map.get(lab.strip(), -1)
        else:
            lab = int(lab)
        if lab in (0, 1):
            clinical_dict[pid] = lab
    return clinical_dict


# 构建文件索引并匹配标签
def build_dataframe(base_dir, clinical_dict):
    # 构建规范化临床字典，去除s-等干扰字符便于后续匹配
    norm_clinical = {}
    for orig_id, label in clinical_dict.items():
        norm_id = re.sub(r'^[sS]-?', '', orig_id)
        norm_clinical[norm_id] = (orig_id, label)

    records = []
    all_folders = sorted(Path(base_dir).glob("Raw Raman spectra data of PCA*"))
    print(f"找到 {len(all_folders)} 个文件夹:")
    for f in all_folders:
        print(f"  - {f.name}")

    for folder in all_folders:
        folder_name = folder.name.lower()
        if "without" in folder_name:
            folder_label = 0
        elif "with" in folder_name:
            folder_label = 1
        else:
            print(f"警告：无法判断文件夹 {folder_name} 的类别，跳过")
            continue

        for txt_file in folder.glob("*.txt"):
            raw_pid = extract_patient_id(txt_file.name)  # 例如 's20140815-02-00309'
            # 对提取的ID做同样的规范化
            norm_pid = re.sub(r'^[sS]-?', '', raw_pid)

            # 匹配
            if norm_pid in norm_clinical:
                orig_id, label = norm_clinical[norm_pid]
                pid = orig_id  # 使用临床表中的原始ID，保证同一患者ID完全一致
            else:
                label = folder_label
                pid = raw_pid

            records.append({
                "filepath": str(txt_file),
                "patient_id": pid,
                "label": label
            })

    df = pd.DataFrame(records)
    print(f"共索引到 {len(df)} 条光谱，来自 {df['patient_id'].nunique()} 名患者")
    return df

# 光谱预处理
def preprocess_spectrum(filepath, common_axis=COMMON_AXIS, sg_window=SG_WINDOW, sg_poly=SG_POLY):
    """读取两列 .txt 文件，执行插值、去尖峰、平滑、基线校正、面积归一化"""
    try:
        # 尝试用 numpy 加载
        data = np.loadtxt(filepath)
    except:
        # 若失败，尝试用 pandas 读取
        data = pd.read_csv(filepath, sep=None, engine='python', header=None).values
    if data.shape[1] < 2:
        raise ValueError(f"文件 {filepath} 不是两列数据")
    wavenum, intensity = data[:, 0], data[:, 1]

    # 插值到公共轴
    interp = interp1d(wavenum, intensity, kind='linear', bounds_error=False, fill_value=0.0)
    spectrum = interp(common_axis)

    # 简单去尖峰
    lower, upper = np.percentile(spectrum, 1), np.percentile(spectrum, 99)
    spectrum = np.clip(spectrum, lower, upper)

    # 平滑
    spectrum = savgol_filter(spectrum, window_length=sg_window, polyorder=sg_poly)

    # 基线校正
    spectrum = spectrum - np.min(spectrum)

    # 面积归一化
    spectrum = (spectrum - np.mean(spectrum)) / (np.std(spectrum) + 1e-8)

    return spectrum


def prepare_data(df, common_axis):
    """对所有光谱进行预处理，返回特征 X、标签 y、患者 ID 列表"""
    X_list, y_list, pid_list = [], [], []
    total = len(df)
    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"  预处理进度: {idx}/{total}")
        spec = preprocess_spectrum(row["filepath"], common_axis)
        X_list.append(spec)
        y_list.append(row["label"])
        pid_list.append(row["patient_id"])
    X = np.array(X_list)
    y = np.array(y_list)
    patient_ids = np.array(pid_list)
    return X, y, patient_ids

# 定义CNN模型
def build_cnn(input_length):
    reg = tf.keras.regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Conv1D(32, 7, activation='relu', padding='same',
                      kernel_regularizer=reg, input_shape=(input_length, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# 交叉验证训练与评估
def run_cross_validation(X, y, patient_ids, n_splits=5, val_patient_ratio=VAL_PATIENT_RATIO):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    fold = 1
    for train_idx, test_idx in sgkf.split(X, y, groups=patient_ids):
        print(f"\n{'=' * 20} Fold {fold}/{n_splits} {'=' * 20}")
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]
        train_val_pids = patient_ids[train_idx]
        test_pids = patient_ids[test_idx]

        # 获取 train_val 中的所有患者及其标签
        unique_pids, pid_indices = np.unique(train_val_pids, return_index=True)
        unique_labels = y_train_val[pid_indices]  # 每个患者的标签

        # 分层划分患者
        train_pids, val_pids = train_test_split(
            unique_pids,
            test_size=val_patient_ratio,
            stratify=unique_labels,
            random_state=RANDOM_STATE
        )

        # 构建训练集和验证集的布尔掩码
        train_mask = np.isin(train_val_pids, train_pids)
        val_mask = np.isin(train_val_pids, val_pids)

        X_train_final = X_train_val[train_mask]
        y_train_final = y_train_val[train_mask]
        X_val_final = X_train_val[val_mask]
        y_val_final = y_train_val[val_mask]

        # 验证集必须有至少两个类别，否则报错
        if len(np.unique(y_val_final)) < 2:
            print("错误：验证集中只有一个类别，请增大 val_patient_ratio 或检查数据！")
            continue

        # 扩展通道维度
        X_train_final = X_train_final[..., np.newaxis]
        X_val_final = X_val_final[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # 构建新模型
        model = build_cnn(input_length=X.shape[1])
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=PATIENCE, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )

        pos_weight = 2.5
        neg_weight = 1.0
        class_weight_dict = {0: neg_weight, 1: pos_weight}

        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # 光谱级评估
        y_prob = model.predict(X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        spec_acc = accuracy_score(y_test, y_pred)
        spec_auc = roc_auc_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # 患者级评估，计算平均概率
        test_df = pd.DataFrame({"patient_id": test_pids, "true": y_test, "prob": y_prob})
        patient_grouped = test_df.groupby("patient_id").agg(
            true_label=("true", "first"),
            avg_prob=("prob", "mean")
        )
        patient_pred_label = (patient_grouped["avg_prob"] > 0.5).astype(int)
        patient_acc = accuracy_score(patient_grouped["true_label"], patient_pred_label)
        patient_auc = roc_auc_score(patient_grouped["true_label"], patient_grouped["avg_prob"])
        tn, fp, fn, tp = confusion_matrix(patient_grouped["true_label"], patient_pred_label).ravel()
        patient_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        patient_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"光谱级 -> Acc: {spec_acc:.4f}, AUC: {spec_auc:.4f}, Sens: {spec_sens:.4f}, Spec: {spec_spec:.4f}")
        print(
            f"患者级 -> Acc: {patient_acc:.4f}, AUC: {patient_auc:.4f}, Sens: {patient_sens:.4f}, Spec: {patient_spec:.4f}")

        fold_results.append({
            "fold": fold,
            "spectral": (spec_acc, spec_auc, spec_sens, spec_spec),
            "patient": (patient_acc, patient_auc, patient_sens, patient_spec),
            "history": history.history
        })
        fold += 1

    return fold_results


# ===================== 7. 主程序 =====================
if __name__ == "__main__":
    print("加载临床信息...")
    clinical_dict = load_clinical_info(EXCEL_PATH)
    print("构建文件索引...")
    df_spectra = build_dataframe(BASE_DIR, clinical_dict)
    print("预处理光谱...")
    X, y, patient_ids = prepare_data(df_spectra, COMMON_AXIS)
    print(f"特征矩阵形状: {X.shape}")
    print("开始交叉验证...")
    results = run_cross_validation(X, y, patient_ids, n_splits=N_SPLITS, val_patient_ratio=VAL_PATIENT_RATIO)

    # 汇总结果
    print("\n" + "=" * 100)
    print("           交叉验证结果汇总（使用最佳阈值）")
    print("=" * 100)
    metrics = ["Accuracy", "AUC", "Sensitivity", "Specificity"]
    print(f"{'':<12}{'光谱级':>40}{'患者级':>40}")
    print(f"{'Metric':<12}" + "".join([f"{m:>10}" for m in metrics * 2]))
    spec_means, patient_means = [], []
    for r in results:
        spec = r["spectral"]
        pat = r["patient"]
        spec_means.append(spec)
        patient_means.append(pat)
        print(f"Fold {r['fold']:<5}" + "".join([f"{v:>10.4f}" for v in spec + pat]))
    spec_avg = np.mean(spec_means, axis=0)
    spec_std = np.std(spec_means, axis=0)
    pat_avg = np.mean(patient_means, axis=0)
    pat_std = np.std(patient_means, axis=0)
    print("-" * 92)
    print(f"{'Mean':<12}" + "".join([f"{v:>10.4f}" for v in np.concatenate([spec_avg, pat_avg])]))
    print(f"{'Std':<12}" + "".join([f"{v:>10.4f}" for v in np.concatenate([spec_std, pat_std])]))

    # 绘制最后一折的训练曲线
    if results:
        last_hist = results[-1]["history"]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(last_hist['loss'], label='Train Loss')
        plt.plot(last_hist['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(last_hist['auc'], label='Train AUC')
        plt.plot(last_hist['val_auc'], label='Val AUC')
        plt.title('AUC')
        plt.legend()
        plt.savefig('optimized_training_curve.png')
        plt.show()
