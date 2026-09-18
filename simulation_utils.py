import numpy as np
from tqdm import tqdm
import math


# 设置标签
def generate_target_spectrum(DOA, doa_min, grid, NUM_GRID):
    K = len(DOA)
    target_vector = 0
    for ki in range(K):
        doa_i = DOA[ki]
        target_vector_i_ = []
        grid_idx = 0
        while grid_idx < NUM_GRID:
            grid_pre = doa_min + grid * grid_idx
            grid_post = doa_min + grid * (grid_idx + 1)
            if grid_pre <= doa_i and grid_post > doa_i:
                expand_vec = np.array([grid_post - doa_i, doa_i - grid_pre]) / grid
                grid_idx += 2
            else:
                expand_vec = np.array([0.0])
                grid_idx += 1
            target_vector_i_.extend(expand_vec)
        if len(target_vector_i_) >= NUM_GRID:
            target_vector_i = target_vector_i_[:NUM_GRID]
        else:
            expand_vec = np.zeros(NUM_GRID - len(target_vector_i_))
            target_vector_i = target_vector_i_
            target_vector_i.extend(expand_vec)
        target_vector += np.asarray(target_vector_i)
    return target_vector


def random_sample(data, data_train, data_val, rate):
    num = len(data['input'])
    threshold = math.floor(10 * rate)
    for temp in range(num):
        if temp % 10 < threshold:
            data_train['input'].append(data['input'][temp])
            data_train['target_spec'].append(data['target_spec'][temp])
        else:
            data_val['input'].append(data['input'][temp])
            data_val['target_spec'].append(data['target_spec'][temp])
    return data_train, data_val


# 生成训练数据（优化：精准SNR校准+功率比独立控制+特征中心化）
def generate_arrays_cov(rate, SNR_ALL, M, Horizontal, power_ratio=1):
    Q = np.eye(M)  # 噪声协方差矩阵
    data = {'input': [], 'target_spec': []}
    data_val = {'input': [], 'target_spec': []}
    data_train = {'input': [], 'target_spec': []}

    derad = np.pi / 180
    wavelength = 0.3  # 波长
    N = 400  # 快拍数
    d = (np.arange(M) * wavelength / 2).reshape(-1, 1)
    doa_max = 60
    doa_min = -60
    step = 1
    NUM_REPEAT = 10
    NUM_GRID = int((doa_max - doa_min) / step)
    doa_delta = np.array(np.arange(120) + 1) * 1  # 角度间隔为1及其倍数

    for SNR in SNR_ALL:
        for delta in tqdm(range(len(doa_delta))):
            delta_curr = doa_delta[delta]
            delta_curr_seq = np.concatenate([[0], [delta_curr]])
            NUM_STEP = int((doa_max - doa_min - delta_curr) / step)

            for step_idx in range(NUM_STEP + 1):
                doa_first = doa_min + step * step_idx
                DOA = delta_curr_seq + doa_first
                y = np.zeros(121)  # 标签
                y[doa_first + 60] = 1  # 修正索引计算
                y[doa_first + delta_curr + 60] = 1

                for rep_idx in range(NUM_REPEAT):
                    # 1. 生成带功率比的信号
                    K = len(DOA)
                    if K == 2:
                        signal_power = np.array([1, power_ratio])
                    elif K == 3:
                        signal_power = np.array([1, power_ratio, power_ratio ** 2])
                    elif K == 4:
                        signal_power = np.array([1, power_ratio, power_ratio ** 2, power_ratio ** 3])
                    else:
                        signal_power = np.ones(K)

                    # 生成复高斯信号
                    S = np.random.randn(K, N) + 1j * np.random.randn(K, N)
                    S = S * np.sqrt(signal_power.reshape(-1, 1))  # 功率比缩放

                    # 2. 生成阵列流形矩阵
                    A = np.exp(-1j * 2 * np.pi * d / wavelength * np.sin(DOA * derad))

                    # 3. 生成纯净信号（无噪声）
                    array_signal = np.matmul(A, S)

                    # 4. 生成噪声并精准校准SNR
                    noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                    noise = np.matmul(Q, noise)  # 噪声协方差加权

                    # 计算当前噪声功率
                    noise_power = np.mean(np.abs(noise) ** 2)
                    # 计算目标信号功率（确保SNR严格符合设定值）
                    target_signal_power = 10 ** (SNR / 10) * noise_power
                    # 计算当前信号功率
                    curr_signal_power = np.mean(np.abs(array_signal) ** 2)
                    # 功率校准因子（避免SNR被稀释）
                    scale_factor = np.sqrt(target_signal_power / curr_signal_power)
                    # 校准后的接收信号
                    received_signal = scale_factor * array_signal + noise

                    # 5. 计算协方差矩阵
                    Rx = np.matmul(received_signal, received_signal.conj().T) / N

                    # 6. 特征提取（实部+虚部）+ 中心化+归一化
                    if M == 16:
                        if Horizontal:
                            uptri_idx = np.triu_indices_from(Rx, k=1)
                            u_real = np.real(Rx[uptri_idx])
                            u_imag = np.imag(Rx[uptri_idx])
                        else:
                            U = np.concatenate([np.diagonal(Rx, offset=i) for i in range(1, 16)])
                            u_real = np.real(U)
                            u_imag = np.imag(U)
                    elif M == 8:
                        if Horizontal:
                            uptri_idx = np.triu_indices_from(Rx, k=1)
                            u_real = np.real(Rx[uptri_idx])
                            u_imag = np.imag(Rx[uptri_idx])
                        else:
                            U = np.concatenate([np.diagonal(Rx, offset=i) for i in range(1, 8)])
                            u_real = np.real(U)
                            u_imag = np.imag(U)
                    else:
                        raise ValueError("仅支持M=8/16")

                    # 关键优化：均值中心化（去除直流分量）
                    u_real = u_real - np.mean(u_real)
                    u_imag = u_imag - np.mean(u_imag)

                    # L2归一化
                    nomal1 = np.linalg.norm(u_real, ord=2) + 1e-8  # 防止除零
                    nomal2 = np.linalg.norm(u_imag, ord=2) + 1e-8
                    u_real = u_real / nomal1
                    u_imag = u_imag / nomal2

                    # 重塑并堆叠特征
                    u_real = u_real.reshape(1, -1)
                    u_imag = u_imag.reshape(1, -1)
                    cov_vector_ext = np.stack((u_real, u_imag), axis=0)

                    data['input'].append(cov_vector_ext)
                    data['target_spec'].append(y)

    print(f"总数据量: {len(data['input'])}")
    data_train, data_val = random_sample(data, data_train, data_val, rate)
    return data_train, data_val


# 生成测试数据（优化：精准SNR+功率比+中心化）
# ═══════════════════════════════════════════════════════════════
# 统一信号生成：所有模型共用同一套 SNR 校准逻辑
# ═══════════════════════════════════════════════════════════════

def _signal_power(K, power_ratio):
    """返回归一化后的功率向量（总功率=K）"""
    if K == 2:
        pwr = np.array([1, power_ratio])
    elif K == 3:
        pwr = np.array([1, power_ratio, power_ratio ** 2])
    elif K == 4:
        pwr = np.array([1, power_ratio, power_ratio ** 2, power_ratio ** 3])
    else:
        pwr = np.ones(K)
    return pwr / pwr.sum() * K


def generate_unified_Rx(DOA, M, N, SNR, wavelength=0.3, power_ratio=1.0, correlation_rho=0.0):
    """统一信号生成 + 精准 SNR 校准 → 返回 Rx。

    所有模型共用此函数生成 Rx，各模型再自行提取特征。
    correlation_rho>0 时注入信源相干（仅 K=2）。
    """
    derad = np.pi / 180
    d = (np.arange(M) * wavelength / 2).reshape(-1, 1)
    K = len(DOA)

    # 1. 信号生成（含功率比）
    pwr = _signal_power(K, power_ratio)
    S = np.random.randn(K, N) + 1j * np.random.randn(K, N)
    S *= np.sqrt(pwr).reshape(-1, 1)

    # 1b. 信源相干注入
    if correlation_rho > 0 and K == 2:
        from scipy.linalg import sqrtm
        Rs = np.array([[1, correlation_rho], [correlation_rho, 1]])
        S = sqrtm(Rs) @ S

    # 2. 阵列流形 + 信号
    A = np.exp(-1j * 2 * np.pi * d / wavelength * np.sin(DOA * derad))
    array_signal = A @ S

    # 3. 噪声（标准复高斯，每阵元方差=2）
    noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)

    # 4. 精准 SNR 校准：实测信号功率 → 回缩到目标 SNR
    noise_power = np.mean(np.abs(noise) ** 2)        # ≈ 2
    target_signal_power = 10 ** (SNR / 10) * noise_power
    curr_signal_power = np.mean(np.abs(array_signal) ** 2)
    scale_factor = np.sqrt(target_signal_power / (curr_signal_power + 1e-12))
    received_signal = scale_factor * array_signal + noise

    # 5. 协方差矩阵
    Rx = (received_signal @ received_signal.conj().T) / N
    return Rx


# ═══════════════════════════════════════════════════════════════
# 各模型特征提取器（输入统一 Rx，输出模型特定格式）
# ═══════════════════════════════════════════════════════════════

def extract_lowsnrcnn_features(Rx, M=16):
    """LowSNR-CNN: 3 通道全矩阵 (real, imag, phase) → (3, M, M)"""
    phase = np.arctan(Rx.imag / Rx.real)  # 和训练时一致，不用 arctan2
    rx_ext = np.stack([Rx.real, Rx.imag, phase], axis=0)
    return {'input': [rx_ext]}


def extract_obcnn_features(Rx, M=16):
    """OBCNN: 对角线拼接 (real+imag) → L2 归一化 → (240,)"""
    diagonals = [np.diagonal(Rx, offset=k) for k in range(1, M)]
    U = np.concatenate(diagonals)
    feat = np.concatenate([np.real(U), np.imag(U)])
    norm = np.linalg.norm(feat, ord=2)
    if norm > 1e-8:
        feat = feat / norm
    else:
        feat = np.zeros_like(feat)
    return {'input': [np.array([feat])]}



def spectral_peak_search(order, P, DOA, doa_min=-60, grid=1.0, WF=True):
    """从 MUSIC/CBF 谱中搜索峰值并匹配真实 DOA。

    order: np.diff(P)
    P: 谱值
    DOA: 真实角度
    doa_min: 网格起始角度
    grid: 网格步长 (度)
    WF: 是否使用加权频率估计进行亚网格插值
    Returns (doa_est, match_errors)
    """
    K = len(DOA)
    N = len(P)
    angles_grid = doa_min + grid * np.arange(N)

    # 1. 找峰值
    peaks = []
    peak_vals = []
    for i in range(1, N - 1):
        if order[i - 1] > 0 and order[i] <= 0 and P[i] > 0:
            peaks.append(i)
            peak_vals.append(P[i])

    # 加边界
    if P[0] > P[1]:
        peaks.insert(0, 0)
        peak_vals.insert(0, P[0])
    if P[-1] > P[-2]:
        peaks.append(N - 1)
        peak_vals.append(P[-1])

    if not peaks:
        return np.full(K, 100.0), np.full(K, 100.0)

    peak_idx = np.array(peaks)
    peak_vals = np.array(peak_vals)
    sort_idx = np.argsort(peak_vals)[::-1]
    peak_idx = peak_idx[sort_idx]
    peak_vals = peak_vals[sort_idx]

    # 亚网格插值
    doa_candidates = []
    for idx in peak_idx:
        if 0 < idx < N - 1:
            a, b, c = P[idx - 1], P[idx], P[idx + 1]
            denom = 2 * (2 * b - a - c)
            if abs(denom) > 1e-12:
                delta = (a - c) / denom
            else:
                delta = 0.0
            doa_fine = angles_grid[idx] + delta * grid
        else:
            doa_fine = angles_grid[idx]
        doa_candidates.append(doa_fine)

    doa_candidates = np.array(doa_candidates)

    # 匹配到最近的真实 DOA
    doa_est = []
    errors = []
    used = set()
    for true_doa in DOA:
        best_err = float('inf')
        best_doa = 100.0
        for j, cand in enumerate(doa_candidates):
            if j in used:
                continue
            err = abs(cand - true_doa)
            if err < best_err:
                best_err = err
                best_doa = cand
                best_j = j
        if best_err < 30.0:  # 30度容忍
            doa_est.append(best_doa)
            errors.append(best_err)
            used.add(best_j)

    # 补足到 K 个
    while len(doa_est) < K:
        doa_est.append(100.0)
        errors.append(100.0)

    return np.sort(np.array(doa_est[:K])), np.array(errors[:K])


def generate_arrays_test_lowsnr(DOA, M, N, SNR, power_ratio=1):
    """LowSNR-CNN 专用信号生成（和训练约定一致，不复用统一 Rx）"""
    Q = np.eye(M)
    data_test_cov = {'input': []}
    derad = np.pi / 180
    wavelength = 0.3
    d = (np.arange(M) * wavelength / 2).reshape(-1, 1)

    K = len(DOA)
    if K == 2:
        signal_power = np.array([1, power_ratio])
    elif K == 3:
        signal_power = np.array([1, power_ratio, power_ratio ** 2])
    elif K == 4:
        signal_power = np.array([1, power_ratio, power_ratio ** 2, power_ratio ** 3])
    else:
        signal_power = np.ones(K)

    S = np.random.randn(K, N) + 1j * np.random.randn(K, N)
    S = S * np.sqrt(signal_power.reshape(-1, 1))

    noise_power = np.sum(Q) / M
    add_noise = 1 / (noise_power * (10 ** (SNR / 10)) ** 0.5) * (
            np.random.randn(M, N) + 1j * np.random.randn(M, N))
    A = np.exp(-1j * 2 * np.pi * d / wavelength * np.sin(DOA * derad))
    array_signal = np.matmul(A, S) + np.matmul(Q, add_noise)
    Rx = np.array(np.matmul(array_signal, np.matrix.getH(array_signal)) / N)
    phase = np.arctan(Rx.imag / Rx.real)

    Rx_vector_ext = np.stack((Rx.real, Rx.imag, phase), axis=0)
    data_test_cov['input'].append(Rx_vector_ext)
    return data_test_cov
