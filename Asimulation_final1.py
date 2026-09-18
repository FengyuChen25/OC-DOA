from simulation_utils import *
from lowcnn_model import *
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from OCDOA_model import OrthoBasisCNN1D
from OCDOA_utils import (
    estimate_angles_from_coefficients_ESPRIT,
)
from hmc_vit_model import cnn_trans
import matplotlib.pyplot as plt
 
plt.rc('font', family='serif', size=10)
plt.rc('mathtext', fontset='stix')
plt.rc('axes', unicode_minus=True)

import os
import time
from datetime import datetime



def main():
    model_config = {
        'run_obcnn': True,
        'run_lowsnrcnn': True,
        'run_music': True,
        'run_cbf': True,
        'run_hmcvit': True
    }


    decision = "phase_error"  # 可选：N/SNR/angle/power_ratio/phase_error/correlation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 自定义信源角度
    DOA = np.array([-34.64,-8.31])
    #DOA = np.array([-34.64, -8.31, 12.45])
    #DOA = np.array([-34.64,-8.31,12.45,28.57])
    #DOA = np.array([-14.65, -7.38])#测相干性用小间隔角度

    K = len(DOA) # 支持2/3/4信源


    base_doa = DOA[0]

    SNR_init = -5
    M = 16
    derad = np.pi / 180
    num_epoch = 100
    wavelength = 0.3
    Num_array = 16
    j1 = 1j
    a = 0
    x_axis4 = np.zeros(Num_array).reshape(Num_array, 1)
    y_axis4 = wavelength / 2 * np.arange(0, Num_array)
    start1 = 0
    delta_a = 0
    delta_p = 0
    d = (np.arange(M) * wavelength / 2).reshape(-1, 1)
    dd = (np.arange(M) * wavelength / 2).reshape(-1, 1)

    # 模型加载
    models = {}
    if model_config['run_obcnn']:
        models['obcnn'] = OrthoBasisCNN1D(input_length=120, output_dim=32).to(device)
        models['obcnn'].load_state_dict(torch.load('ocdoa/best_model_obmodel1.pth', map_location=device))
        #_, _, base_info = load_orthobasis_data("orthobasis_data_M16")
        base_info = np.load(os.path.join("orthobasis_data_M16", 'base_info.npy'), allow_pickle=True).item()
        models['obcnn_base_info'] = base_info

    if model_config['run_lowsnrcnn']:
        models['lowsnrcnn'] = lowsnr_cnn().to(device)
        models['lowsnrcnn'].load_state_dict(torch.load('lowsnrcnn/net.pth', map_location=device))

    if model_config['run_hmcvit']:  # HMC-ViT
        models['hmcvit'] = cnn_trans(in_channels=3, patch_size_x=1, patch_size_y=6,
                                      emb_size=128, img_size=120, depth=4, num_cls_tokens=8).to(device)
        ckpt = torch.load("hmcvit/best.pth", map_location=device, weights_only=True)
        models['hmcvit'].load_state_dict(ckpt.get('model_state', ckpt))
        models['hmcvit'].eval()
        models['hmcvit_grid'] = np.linspace(-60, 60, 121)

    # 实验参数配置
    if decision == "N":
        Rho = [30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000]
        plt.xlabel('Number of snapshots', fontsize=9)
    elif decision == "SNR":
        Rho = np.linspace(-14, 6, 11)
        plt.xlabel('SNR(dB)', fontsize=9)
    elif decision == "angle":
        Rho = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        plt.xlabel('DOA separation(degree)', fontsize=9)
    elif decision == "power_ratio":
        Rho =  [1 + 1*i for i in range(8)]  # 2信源：信号2功率/信号1功率
        plt.xlabel('Power ratio (signal2/signal1)', fontsize=9)
    elif decision == "phase_error":
        Rho = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40]  # 相位误差标准差 σ (degree)
        plt.xlabel(r'Phase Error $\sigma_{\varphi}$ (degree)', fontsize=9)
    elif decision == "correlation":
        Rho = np.arange(0, 1.1, 0.1)
        plt.xlabel('Source Correlation Coefficient $\\rho$', fontsize=9)

# RMSE初始化
    RMSE_dict = {
        'obcnn': [], 'lowsnrcnn': [], 'hmcvit': [],
        'MUSIC': [], 'MUSIC_01': [], 'CBF': [], 'CBF_01': [], 'crb': []
    }

    t0 = time.time()
    ts_prefix = datetime.now().strftime("%y%m%d%H%M")  # e.g., 2608060933
    phase_error_flag = False; delta_p_deg = 0
    correlation_flag = False; correlation_rho = 0
    amp_error_flag = False; amp_error_delta = 0
    A_amp = np.eye(M); P_err = np.eye(M)
    for rho_idx, rho in enumerate(Rho):
        # 参数赋值
        if decision == "N":
            N = int(rho)
            D_num1 = N
            SNR = SNR_init
            power_ratio=1
            file_plt = f'./pic_final/{ts_prefix}_N_K{K}_{DOA[0]}_{DOA[1]}_SNR{SNR}.pdf'
        elif decision == "SNR":
            N = 400
            D_num1 = N
            SNR = int(rho)
            power_ratio=1
            file_plt = f'./pic_final/{ts_prefix}_SNR_K{K}_{DOA[0]}_{DOA[1]}_N{N}.pdf'
        elif decision == "angle":
            N = 400
            D_num1 = N
            SNR = SNR_init
            power_ratio=1
            if K == 2:
                DOA[1] = base_doa + rho
            if K == 3:
                DOA[2] = base_doa + rho
            if K == 4:
                DOA[3] = base_doa + rho
            file_plt = f'./pic_final/{ts_prefix}_ANGLE_K{K}_FirstDOA{DOA[0]}_SNR{SNR_init}_N{N}.pdf'
        elif decision == "power_ratio":
            N = 400
            D_num1 = N
            SNR = SNR_init
            power_ratio = rho
            file_plt = f'./pic_final/{ts_prefix}_POWER_RATIO_K{K}_SNR{SNR_init}_N{N}.pdf'
        elif decision == "phase_error":
            N = 400
            D_num1 = N
            SNR = SNR_init
            power_ratio = 1
            delta_p_deg = float(rho)  # rho = σ (degrees)
            delta_p = delta_p_deg  # 接线: CRB 的相位误差模型用同一参数
            phase_error_flag = True
            file_plt = f'./pic_final/{ts_prefix}_PHASE_ERR_{DOA[0]}_{DOA[1]}_SNR{SNR}_N{N}.pdf'
        elif decision == "correlation":
            N = 400; D_num1 = N
            SNR = SNR_init; power_ratio = 1
            correlation_rho = float(rho)
            correlation_flag = True
            file_plt = f'./pic_final/{ts_prefix}_CORR_{DOA[0]}_{DOA[1]}_SNR{SNR}_N{N}.pdf'
        else:
            phase_error_flag = False; delta_p_deg = 0
            correlation_flag = False; correlation_rho = 0
            amp_error_flag = False; amp_error_delta = 0

        MSE_dict = {key: np.zeros(K, ) for key in RMSE_dict.keys() if key != 'crb'}

        np.random.seed(725)
        for epoch in range(num_epoch):
            epoch_est = {
                'true': np.sort(DOA.copy()),
                'obcnn': None,
                'music01': None
            }

            # ═══════════════════════════════════════
            # 统一信号生成 (generate_unified_Rx) + 统一特征提取
            # ═══════════════════════════════════════
            current_power_ratio = power_ratio if decision == "power_ratio" else 1.0

            # 生成统一 Rx（所有模型共享）
            need_Rx = ( model_config['run_music'] or
                       model_config['run_cbf'] or model_config['run_hmcvit'] or
                       model_config['run_lowsnrcnn'] or model_config['run_obcnn'])
            if need_Rx:
                Rx = generate_unified_Rx(DOA, M, N, SNR, wavelength=0.3,
                                         power_ratio=current_power_ratio,
                                         correlation_rho=correlation_rho if correlation_flag else 0.0)

            # 从统一 Rx 提取各模型特征

            if model_config['run_lowsnrcnn']:
                data_lowsnrcnn = extract_lowsnrcnn_features(Rx, M)
                input_array = np.array(data_lowsnrcnn['input'])
                data_tensor_lowsnrcnn = torch.tensor(input_array, dtype=torch.float32)

            if model_config['run_obcnn']:
                data_obcnn = extract_obcnn_features(Rx, M)
                input_obcnn = torch.from_numpy(np.array(data_obcnn['input'])).float()
                input_obcnn = input_obcnn.reshape(-1, 2, 1, 120).to(device)

            # ═══════════════════════════════════════
            # Perturbation: 相位误差 → 污染统一 Rx → 重提取特征
            # 相干已在 generate_unified_Rx 中注入
            # ═══════════════════════════════════════

            if phase_error_flag and delta_p_deg > 0:
                # 论文(42): φ_m = √12·σ·η_m, η_m~U[-0.5,0.5] (均匀, std=σ)
                eta = np.random.rand(M) - 0.5
                phase_err_deg = np.sqrt(12) * delta_p_deg * eta
                P_err = np.diag(np.exp(1j * phase_err_deg * np.pi / 180))
                Rx = P_err @ Rx @ P_err.conj().T
                if model_config['run_lowsnrcnn']:
                    data_lowsnrcnn = extract_lowsnrcnn_features(Rx, M)
                    data_tensor_lowsnrcnn = torch.tensor(np.array(data_lowsnrcnn['input']), dtype=torch.float32)
                if model_config['run_obcnn']:
                    data_obcnn = extract_obcnn_features(Rx, M)
                    input_obcnn = torch.from_numpy(np.array(data_obcnn['input'])).float().reshape(-1, 2, 1, 120).to(device)

            # MUSIC计算
            if model_config['run_music']:
                ev, I = np.linalg.eig(Rx)
                sorted_indices = np.argsort(ev)  # 升序: [0]最小特征值, [-1]最大
                U = np.asmatrix(I[:, sorted_indices[:-K]])  # 噪声子空间 = M-K 个最小特征向量

                # MUSIC 1°网格
                P_MUSIC = np.zeros(120)
                for i in range(120):
                    doa = 1 * i - 60
                    a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin(doa * derad))
                    P_MUSIC[i] = (1 / np.abs(np.matrix.getH(a) @ U @ U.H @ a)).item()
                P_MUSIC = P_MUSIC / np.max(P_MUSIC)
                order_music = np.diff(P_MUSIC)
                doa_music, _ = spectral_peak_search(order_music, P_MUSIC, DOA, doa_min=-60, grid=1, WF=True)
                doa_music = np.sort(np.array(doa_music))
                
                if len(doa_music) < K:
                    doa_music = np.pad(doa_music, (0, K - len(doa_music)), mode='constant', constant_values=0)
                elif len(doa_music) > K:
                    doa_music = doa_music[:K]
                MSE_dict['MUSIC'] += np.square(doa_music - DOA)

                # MUSIC 0.1°网格
                P_MUSIC_01 = np.zeros(1200)
                for i in range(1200):
                    doa = 0.1 * i - 60
                    a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin(doa * derad))
                    P_MUSIC_01[i] = (1 / np.abs(np.matrix.getH(a) @ U @ U.H @ a)).item()
                P_MUSIC_01 = P_MUSIC_01 / np.max(P_MUSIC_01)
                order_music_01 = np.diff(P_MUSIC_01)
                doa_music_01, _ = spectral_peak_search(order_music_01, P_MUSIC_01, DOA, doa_min=-60, grid=0.1, WF=True)
                doa_music_01 = np.sort(np.array(doa_music_01))
                if len(doa_music_01) < K:
                    doa_music_01 = np.pad(doa_music_01, (0, K - len(doa_music_01)), mode='constant', constant_values=0)
                elif len(doa_music_01) > K:
                    doa_music_01 = doa_music_01[:K]
                MSE_dict['MUSIC_01'] += np.square(doa_music_01 - DOA)
                epoch_est['music01'] = doa_music_01.copy()

                # ——— MUSIC spectrum  first epoch only ———
                if decision == "correlation" and epoch == 0 and rho==0 and rho==0.5 and rho==1:
                    fig_s, ax_s = plt.subplots(figsize=(8, 4))
                    theta_01 = np.arange(-60, 60, 0.1)
                    ax_s.plot(theta_01, P_MUSIC_01, 'b-', lw=1.2, label='MUSIC 0.1deg')
                    for d in DOA:
                        ax_s.axvline(d, color='green', ls=':', lw=1.5)
                    for d in doa_music_01:
                        ax_s.axvline(d, color='orange', ls='--', lw=1)
                    ax_s.set_xlabel('DOA (deg)'); ax_s.set_ylabel('Normalized spectrum')
                    ax_s.set_title(f'MUSIC spectrum rho={rho} SNR={SNR}dB')
                    ax_s.legend(fontsize=7); ax_s.grid(True, alpha=0.3)
                    fig_s.tight_layout()
                    os.makedirs('pic_final', exist_ok=True)
                    fig_s.savefig(f'pic_final/music_spectrum_rho_{rho}.pdf', dpi=150, bbox_inches='tight')
                    plt.close(fig_s)
                    print(f"  [MUSIC spectrum saved]")

            # CBF计算
            if model_config['run_cbf']:
                # CBF 1°网格
                P_CBF = np.zeros(120)
                for i in range(120):
                    doa = 1 * i - 60
                    a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin(doa * derad))
                    P_CBF[i] = np.abs(np.matrix.getH(a) @ Rx @ a).item()
                P_CBF = P_CBF / np.max(P_CBF)
                order_cbf = np.diff(P_CBF)
                doa_cbf, _ = spectral_peak_search(order_cbf, P_CBF, DOA, doa_min=-60, grid=1, WF=True)
                doa_cbf = np.sort(np.array(doa_cbf))
                if len(doa_cbf) < K:
                    doa_cbf = np.pad(doa_cbf, (0, K - len(doa_cbf)), mode='constant', constant_values=0)
                elif len(doa_cbf) > K:
                    doa_cbf = doa_cbf[:K]
                MSE_dict['CBF'] += np.square(doa_cbf - DOA)

                # CBF 0.1°网格
                P_CBF_01 = np.zeros(1200)
                for i in range(1200):
                    doa = 0.1 * i - 60
                    a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin(doa * derad))
                    P_CBF_01[i] = np.abs(np.matrix.getH(a) @ Rx @ a).item()
                P_CBF_01 = P_CBF_01 / np.max(P_CBF_01)
                order_cbf_01 = np.diff(P_CBF_01)
                doa_cbf_01, _ = spectral_peak_search(order_cbf_01, P_CBF_01, DOA, doa_min=-60, grid=0.1, WF=True)
                doa_cbf_01 = np.sort(np.array(doa_cbf_01))
                if len(doa_cbf_01) < K:
                    doa_cbf_01 = np.pad(doa_cbf_01, (0, K - len(doa_cbf_01)), mode='constant', constant_values=0)
                elif len(doa_cbf_01) > K:
                    doa_cbf_01 = doa_cbf_01[:K]
                MSE_dict['CBF_01'] += np.square(doa_cbf_01 - DOA)



            # HMC-ViT评估
            if model_config['run_hmcvit']:
                with torch.no_grad():
                    tr = np.trace(np.abs(Rx)) + 1e-12
                    Rx_n = Rx * (M / tr)
                    hmc_in = np.stack([Rx_n.real, Rx_n.imag,
                                        np.arctan2(Rx_n.imag, Rx_n.real)], axis=0)
                    idx_ut = np.triu_indices(M, k=1)
                    hmc_in = hmc_in[:, idx_ut[0], idx_ut[1]]
                    hmc_in_t = torch.from_numpy(hmc_in).float().unsqueeze(0).unsqueeze(2).to(device)
                    hmc_out = torch.sigmoid(models['hmcvit'](hmc_in_t)).detach().cpu().numpy().flatten()
                    # 局部极大值选峰（相邻bin不能同时入选），不足K个时回退argsort
                    peaks, _ = find_peaks(hmc_out)
                    if len(peaks) < K:
                        peaks = np.argsort(hmc_out)[::-1][:K]
                    hmc_peaks = sorted(peaks, key=lambda x: hmc_out[x], reverse=True)[:K]
                    hmc_est = np.sort(models['hmcvit_grid'][hmc_peaks])
                    if len(hmc_est) < K:
                        hmc_est = np.pad(hmc_est, (0, K - len(hmc_est)), mode='constant')
                    MSE_dict['hmcvit'] += np.square(hmc_est[:K] - DOA)

            # LowSNR-CNN评估
            if model_config['run_lowsnrcnn']:
                models['lowsnrcnn'].eval()
                with torch.no_grad():
                    result_lowsnrcnn = models['lowsnrcnn'](data_tensor_lowsnrcnn.to(device))
                    result_lowsnrcnn = result_lowsnrcnn.cpu()
                    P_result_lowsnrcnn = result_lowsnrcnn.numpy().squeeze()
                    # 局部极大值选峰（相邻bin不能同时入选）
                    peaks, _ = find_peaks(P_result_lowsnrcnn)
                    if len(peaks) < K:
                        peaks = np.argsort(P_result_lowsnrcnn)[::-1][:K]
                    max_indices = sorted(peaks, key=lambda x: P_result_lowsnrcnn[x], reverse=True)[:K]
                    doa_est = [p - 60 for p in max_indices]
                    doa_est = np.sort(doa_est)
                    if len(doa_est) < K:
                        doa_est = np.pad(doa_est, (0, K - len(doa_est)), mode='constant')
                    MSE_dict['lowsnrcnn'] += np.square(np.array(doa_est) - DOA)
                    epoch_est['lowsnrcnn'] = doa_est.copy()


            #obcnn评估
            if model_config['run_obcnn']:
                models['obcnn'].eval()
                with torch.no_grad():
                    coeff_pred = models['obcnn'](input_obcnn)
                    coeff_pred = coeff_pred.cpu().numpy().squeeze()


                    estimated_angles, _, magnitudes_norm = estimate_angles_from_coefficients_ESPRIT(
                        coeff_pred, models['obcnn_base_info'], max_sources=K
                    )

                    # 提取前K个角度
                    doa_est_obcnn = estimated_angles[:K]
                    # 角度匹配：匈牙利算法
                    true_angles_sorted = np.sort(DOA)
                    cost_matrix = np.abs(doa_est_obcnn.reshape(-1, 1) - true_angles_sorted.reshape(1, -1))
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    obcnn_errors = np.zeros(K)
                    for i, (r, c) in enumerate(zip(row_ind, col_ind)):
                        if i < K:
                            obcnn_errors[i] = cost_matrix[r, c]

                    # MSE计算
                    mse_obcnn = np.square(obcnn_errors)
                    MSE_dict['obcnn'] += mse_obcnn
                    epoch_est['obcnn'] = doa_est_obcnn.copy()

            # 打印中间结果
            if epoch % 10 == 0:
                print(f"\n===== 实验参数：{decision}={rho} | 蒙特卡洛轮次 {epoch + 1}/{num_epoch} =====")
                print(f"真实角度 (排序后): {np.round(epoch_est['true'], 2)}°")
                if decision == "power_ratio":
                    print(f"当前功率比: {power_ratio} (signal2/signal1)")  # 修正标注

                if model_config['run_obcnn'] and epoch_est['obcnn'] is not None:
                    obcnn_errors = np.abs(epoch_est['obcnn'] - epoch_est['true'])
                    avg_error = np.mean(obcnn_errors)
                    print(f"OBCNN估计角度: {np.round(epoch_est['obcnn'], 2)}°")
                    print(f"OBCNN匹配误差: {np.round(obcnn_errors, 2)}° (平均: {np.round(avg_error, 2)}°)")



                if model_config['run_lowsnrcnn'] and epoch_est.get('lowsnrcnn') is not None:
                    low_errors = np.abs(epoch_est['lowsnrcnn'] - epoch_est['true'])
                    avg_low_error = np.mean(low_errors)
                    print(f"CNN(LowSNR)估计角度: {np.round(epoch_est['lowsnrcnn'], 2)}°")
                    print(f"CNN(LowSNR)匹配误差: {np.round(low_errors, 2)}° (平均: {np.round(avg_low_error, 2)}°)")


                if model_config['run_music'] and epoch_est['music01'] is not None:
                    music01_errors = np.abs(epoch_est['music01'] - epoch_est['true'])
                    avg_music01_error = np.mean(music01_errors)
                    print(f"MUSIC(0.1°)估计角度: {np.round(epoch_est['music01'], 2)}°")
                    print(f"MUSIC(0.1°)匹配误差: {np.round(music01_errors, 2)}° (平均: {np.round(avg_music01_error, 2)}°)")

                print("-" * 80)

        # CRB计算
        steering = np.zeros((Num_array, K), dtype=np.complex128)
        steering_deriv = np.zeros((Num_array, K), dtype=np.complex128)
        for J in range(K):
            for I in range(Num_array):
                d_pos = x_axis4[I] * np.cos(DOA[J] * np.pi / 180) + y_axis4[I] * np.sin(DOA[J] * np.pi / 180)
                steering[I, J] = np.exp(-j1 * np.pi * 2 * d_pos / wavelength).item()
                steering_deriv[I, J] = (
                    np.exp(-j1 * np.pi * 2 * d_pos / wavelength) * (-j1 * np.pi * 2 * 1 / wavelength) *
                    (-x_axis4[I] * np.sin(DOA[J] * np.pi / 180) + y_axis4[I] * np.cos(DOA[J] * np.pi / 180))
                ).item()

        np.random.seed(0)
        A_error = np.ones(Num_array).reshape(M, 1)
        A_error = A_error + np.sqrt(12) * (np.random.rand(Num_array).reshape(M, 1) - 0.5) * delta_a
        A_error[0] = 1
        P_error0 = np.sqrt(12) * (np.random.rand(Num_array).reshape(M, 1) - 0.5) * delta_p
        P_error0[0] = 0

        xx = np.multiply(A_error, np.exp(1j * np.pi / 180 * P_error0))
        AP_error = np.diag(np.asarray(xx).ravel())
        steering_Err = np.dot(AP_error, steering)
        n_power_prior0 = 1
        n_power_prior1 = 1 / (10 ** (SNR / 10)) * n_power_prior0

        # 统一功率比适配
        if K == 2:
            signal_power = np.array([1, current_power_ratio])
        elif K == 3:
            signal_power = np.array([1, current_power_ratio, current_power_ratio**2])
        elif K == 4:
            signal_power = np.array([1, current_power_ratio, current_power_ratio**2, current_power_ratio**3])
        else:
            signal_power = np.ones(K)
        # 归一化总功率
        signal_power = signal_power / signal_power.sum()  # 总功率归一化到1，SNR对齐统一生成
        Rs = np.diag(signal_power)
        # Inject source correlation into CRB when applicable
        if decision == "correlation" and correlation_rho > 0:
            for i_ in range(K):
                for j_ in range(K):
                    if i_ != j_:
                        Rs[i_, j_] = correlation_rho * np.sqrt(signal_power[i_] * signal_power[j_])

        R = np.dot(np.dot(steering_Err, Rs), np.conj(steering_Err.T)) + n_power_prior1 * np.eye(Num_array)
        B = np.dot(AP_error[start1:, start1:], steering[start1:Num_array, :])
        Y = np.dot(np.dot(Rs, np.conj(B.T)), np.linalg.pinv(R)).dot(B).dot(Rs)
        D_theta = np.dot(AP_error[start1:, start1:], steering_deriv[start1:, :])
        PB_orth = np.eye(Num_array - start1) - np.dot(B, np.dot(np.linalg.pinv(np.dot(np.conj(B.T), B)), np.conj(B.T)))
        J_thetaTheta = np.multiply(np.dot(np.dot(np.conj(D_theta.T), PB_orth), D_theta), Y.T)
        zz = np.linalg.pinv(2 * D_num1 / n_power_prior1 * np.real(J_thetaTheta))
        DOA_CRB01 = np.sqrt(np.mean(np.diag(np.asarray(zz)))) * 180 / np.pi
        RMSE_dict['crb'].append(DOA_CRB01)

        # 计算RMSE
        for key in RMSE_dict.keys():
            if key == 'crb':
                continue
            if key == 'hmcvit':
                if model_config['run_hmcvit']:
                    rmse = np.sqrt(np.sum(MSE_dict[key]) / (num_epoch * K))
                    RMSE_dict[key].append(rmse)
                continue
            if key == 'hmcvit':
                if model_config['run_hmcvit']:
                    rmse = np.sqrt(np.sum(MSE_dict[key]) / (num_epoch * K))
                    RMSE_dict[key].append(rmse)
                continue
            base_key = key.replace("_01", "").replace("_peaks", "").lower()
            if not model_config.get(f'run_{base_key}', False):
                continue
            rmse = np.sqrt(np.sum(MSE_dict[key]) / (num_epoch * K))
            RMSE_dict[key].append(rmse)

        # 打印RMSE汇总
        print(f"\n===== {decision}={rho} =====")
        print(f"  OBCNN={RMSE_dict['obcnn'][-1]:.4f}" if model_config['run_obcnn'] else "", end="")

        print(f"  CNN={RMSE_dict['lowsnrcnn'][-1]:.4f}" if model_config['run_lowsnrcnn'] else "", end="")
        print(f"  MUSIC_1={RMSE_dict['MUSIC'][-1]:.4f}" if model_config['run_music'] else "", end="")
        print(f"  MUSIC_01={RMSE_dict['MUSIC_01'][-1]:.4f}" if model_config['run_music'] else "", end="")
        print(f"  CBF_1={RMSE_dict['CBF'][-1]:.4f}" if model_config['run_cbf'] else "", end="")
        print(f"  CBF_01={RMSE_dict['CBF_01'][-1]:.4f}" if model_config['run_cbf'] else "")
        print(f"  HMCViT={RMSE_dict['hmcvit'][-1]:.4f}" if model_config['run_hmcvit'] else "")

    # 绘图
    plt.ylabel('RMSE of DOA estimates(degree)', fontsize=10)
    plt.ylim(10 ** (-2), 10 ** 2)
    plt.yscale('log')
    x_uniform = np.linspace(min(Rho), max(Rho), len(Rho))

    plot_config = [
        ('obcnn', 'purple', 'P', '-', 'OC-DOA'),
        ('lowsnrcnn', 'blue', 's', '-', 'CNN'),
        ('MUSIC', 'green', 'D', '-', 'MUSIC with grid 1°'),
        ('MUSIC_01', 'green', 'D', '--', 'MUSIC with grid 0.1°'),
        ('CBF', 'grey', '*', '-', 'CBF with grid 1°'),
        ('CBF_01', 'grey', '*', '--', 'CBF with grid 0.1°'),
        ('hmcvit', '#c41e3a', 'v', '--', 'HMC‑ViT'),
        ('crb', 'black', 'X', '--', 'CRB')
    ]

    for key, color, marker, linestyle, label in plot_config:
        if key not in ('crb', 'hmcvit'):
            base_key = key.replace("_01", "").replace("_peaks", "").lower()
            if not model_config.get(f'run_{base_key}', False):
                continue
        plt.plot(
            x_uniform,
            RMSE_dict[key],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.6,
            markersize=5.5,
            label=label
        )

    plt.grid(linestyle='--')
    plt.tight_layout()
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1, 1), framealpha=0.5, fontsize=10)
    # 可选：设置图例背景为半透明
    legend.get_frame().set_alpha(0.5)

    if decision in ("power_ratio", "correlation"):
        plt.xticks(x_uniform, [f'{x:.2f}' for x in Rho])
    else:
        plt.xticks(x_uniform, [f'{int(x)}' for x in Rho])

    os.makedirs('pic_final', exist_ok=True)
    plt.savefig(file_plt, dpi=300, bbox_inches='tight')
    print(f"\n图形已保存到: {file_plt}")

    # Save RMSE table to CSV
    csv_path = f'./pic_final/{ts_prefix}_{decision}_RMSE.csv'
    # Build transposed table: rows=methods, columns=rho values
    methods = []
    methods.append(('CRB', RMSE_dict['crb']))
    if model_config['run_obcnn']: methods.append(('OBCNN', RMSE_dict['obcnn']))
    if model_config['run_lowsnrcnn']: methods.append(('CNN', RMSE_dict['lowsnrcnn']))
    if model_config['run_hmcvit']: methods.append(('HMCViT', RMSE_dict['hmcvit']))
    if model_config['run_music']:
        methods.append(('MUSIC_1deg', RMSE_dict['MUSIC']))
        methods.append(('MUSIC_01deg', RMSE_dict['MUSIC_01']))
    if model_config['run_cbf']:
        methods.append(('CBF_1deg', RMSE_dict['CBF']))
        methods.append(('CBF_01deg', RMSE_dict['CBF_01']))
    with open(csv_path, 'w') as f:
        header = f'decision = {decision}/rho,' + ','.join([f'{r:.4f}' for r in Rho])
        f.write(header + '\n')
        for name, data in methods:
            f.write(name + ',' + ','.join([f'{v:.4f}' for v in data]) + '\n')
    print(f"RMSE表格已保存到: {csv_path}")
    print(f"总运行时间: {time.time() - t0:.2f}秒")


if __name__ == '__main__':
    main()