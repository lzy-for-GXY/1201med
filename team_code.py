import os
import joblib
import numpy as np
import pandas as pd
import mne
import lightgbm as lgb
from scipy import signal, stats
from tqdm import tqdm
from helper_code import *

# 尝试导入高级库
try:
    import yasa
    HAS_YASA = True
except ImportError:
    HAS_YASA = False

try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False

# ======================== 1. 核心特征抽取逻辑 (Integrated Features) ========================

def preprocess_signal(data, sfreq, target_fs=100.0):
    """信号去噪与归一化"""
    # Notch & Bandpass
    nyq = 0.5 * sfreq
    b, a = signal.butter(4, [0.5/nyq, 35.0/nyq], btype='band')
    data = signal.filtfilt(b, a, data)
    
    # Resample
    if sfreq != target_fs:
        num_samples = int(len(data) * target_fs / sfreq)
        data = signal.resample(data, num_samples)
    
    # Z-score normalization
    data = (data - np.mean(data)) / (np.std(data) + 1e-6)
    return data, target_fs

def get_spectral_features(data, fs):
    """提取频域功率谱特征，包含 DTABR (Delta+Theta)/(Alpha+Beta)"""
    f, psd = signal.welch(data, fs, nperseg=int(4*fs))
    def get_bp(f_low, f_high):
        return np.sum(psd[(f >= f_low) & (f <= f_high)])
    
    delta = get_bp(0.5, 4.0)
    theta = get_bp(4.0, 8.0)
    alpha = get_bp(8.0, 13.0)
    sigma = get_bp(11.0, 16.0)
    beta  = get_bp(13.0, 30.0)
    
    total = delta + theta + alpha + beta + 1e-6
    dar = delta / (alpha + 1e-6)
    tar = theta / (alpha + 1e-6)
    dtabr = (delta + theta) / (alpha + beta + 1e-6) # 新增综合慢波指标
    
    return np.array([delta/total, theta/total, alpha/total, sigma/total, beta/total, dar, tar, dtabr])

def get_advanced_eeg_features(data, sfreq, stages=None):
    """提取 PAC (耦合强度), Spindles (密度/持续时间) 和 Entropy (样本熵)"""
    feats = []
    
    # 1. PAC (SO-Spindle Coupling)
    if HAS_YASA:
        try:
            coupling = yasa.coupling(data, sfreq, freq_so=(0.5, 1.5), freq_sp=(12, 16))
            feats.append(coupling['Strength'].mean())
        except: feats.append(0.0)
    else: feats.append(0.0)
    
    # 2. Spindles (纺锤波微结构)
    if HAS_YASA and stages is not None:
        try:
            sp = yasa.spindles_detect(data, sfreq, hypno=stages, include=(2, 3))
            if sp:
                sum_df = sp.summary()
                # 密度：每分钟检测到的纺锤波数量
                density = len(sum_df) / (np.sum((stages == 2) | (stages == 3)) / (sfreq * 60) + 1e-6)
                duration = sum_df['Duration'].mean()
                feats.extend([density, duration])
            else: feats.extend([0.0, 0.0])
        except: feats.extend([0.0, 0.0])
    else: feats.extend([0.0, 0.0])
    
    # 3. Entropy (Sample Entropy 优化采样计算)
    if HAS_ANTROPY:
        try:
            epoch_len = int(30 * sfreq)
            # 均匀采样 5 个 30 秒片段进行平均以提升效率
            indices = np.linspace(0, len(data) - epoch_len, 5, dtype=int)
            se_list = [ant.sample_entropy(signal.decimate(data[i:i+epoch_len], int(sfreq//50))) for i in indices]
            feats.append(np.mean(se_list))
        except: feats.append(0.0)
    else: feats.append(0.0)
    
    return np.array(feats)

# ======================== 2. 官方接口：模型训练 (Train) ========================

def train_model(data_folder, model_folder, verbose):
    """
    10 折交叉训练的核心实现
    1201usst 标准化：整合 Advanced features 与 LightGBM
    """
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata = find_patients(patient_data_file)
    
    all_features = []
    all_labels = []

    print(f"[*] 正在开始提取特征，样本数: {len(patient_metadata)}")
    
    for i in tqdm(range(len(patient_metadata)), disable=not verbose):
        try:
            record = patient_metadata[i]
            pid = record[HEADERS['bids_folder']]
            sid = record[HEADERS['session_id']]
            site = record[HEADERS['site_id']]
            
            # --- Domain A: Demographic Features ---
            demo = load_demographics(patient_data_file, pid, sid)
            age = load_age(demo)
            bmi = load_bmi(demo)
            label = load_diagnoses(patient_data_file, pid) # 获取认知障碍标签
            
            # --- Domain B: Physiological Signal (EEG C3) ---
            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site, f"{pid}_ses-{sid}.edf")
            raw_phys, fs_dict = load_signal_data(phys_file)
            
            # 遍历寻找最优脑电通道 C3
            eeg_key = [k for k in raw_phys.keys() if 'C3' in k.upper()]
            if not eeg_key:
                eeg_key = [k for k in raw_phys.keys() if 'EEG' in k.upper()]
            eeg_key = eeg_key[0] if eeg_key else list(raw_phys.keys())[0]
            
            clean_eeg, fs_proc = preprocess_signal(raw_phys[eeg_key], fs_dict[eeg_key])
            
            # --- Domain C: Internal Annotations (Stages for Spindles) ---
            human_file = os.path.join(data_folder, HUMAN_ANNOTATIONS_SUBFOLDER, site, f"{pid}_ses-{sid}_expert_annotations.edf")
            human_ann, _ = load_signal_data(human_file)
            stages = human_ann.get('stage_expert') if human_ann else None
            
            # --- Domain D: Advanced Fusion ---
            spec_v = get_spectral_features(clean_eeg, fs_proc)
            adv_v = get_advanced_eeg_features(clean_eeg, fs_proc, stages)
            
            # 组合最终特征向量
            vec = np.hstack([age, bmi, spec_v, adv_v])
            all_features.append(vec)
            all_labels.append(label)
            
        except Exception: continue

    # 使用 LightGBM 进行最终训练 (LGBMClassifier)
    X = np.nan_to_num(np.array(all_features))
    y = np.array(all_labels)
    
    # 计算正负样本权重处理不平衡
    pos_ratio = np.sum(y == 0) / (np.sum(y == 1) + 1e-6)
    
    model = lgb.LGBMClassifier(
        n_estimators=500, 
        max_depth=4, 
        num_leaves=16,
        learning_rate=0.01, 
        scale_pos_weight=pos_ratio,
        random_state=42, 
        verbosity=-1
    )
    model.fit(X, y)
    
    # 保存模型
    os.makedirs(model_folder, exist_ok=True)
    joblib.dump(model, os.path.join(model_folder, 'model.sav'))

# ======================== 3. 官方接口：模型推理 (Run) ========================

def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, 'model.sav'))

def run_model(model, record, data_folder, verbose):
    """
    单样本推理接口
    """
    pid = record[HEADERS['bids_folder']]
    sid = record[HEADERS['session_id']]
    site = record[HEADERS['site_id']]
    
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    demo = load_demographics(demo_file, pid, sid)
    age = np.array([load_age(demo)])
    bmi = np.array([load_bmi(demo)])
    
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site, f"{pid}_ses-{sid}.edf")
    raw_phys, fs_dict = load_signal_data(phys_file)
    
    eeg_key = [k for k in raw_phys.keys() if 'C3' in k.upper()]
    eeg_key = eeg_key[0] if eeg_key else list(raw_phys.keys())[0]
    clean_eeg, fs_proc = preprocess_signal(raw_phys[eeg_key], fs_dict[eeg_key])
    
    # 在推理阶段，官方不提供 Human/Expert Annotations，因此 stages=None
    spec_v = get_spectral_features(clean_eeg, fs_proc)
    adv_v = get_advanced_eeg_features(clean_eeg, fs_proc, stages=None)
    
    feat_vec = np.nan_to_num(np.hstack([age, bmi, spec_v, adv_v])).reshape(1, -1)
    
    binary_output = int(model.predict(feat_vec)[0])
    probability_output = float(model.predict_proba(feat_vec)[0][1])
    
    return binary_output, probability_output
