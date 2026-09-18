import joblib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
def save_orthobasis_data(data_train, data_val, Q, sin_theta_grid, theta_grid, theta_grid_deg, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 使用 joblib 保存训练和验证数据，指定 protocol=5（Python 3.8+），并启用压缩
    joblib.dump(data_train, os.path.join(save_dir, 'train_data.joblib'), compress=True, protocol=5)
    joblib.dump(data_val, os.path.join(save_dir, 'val_data.joblib'), compress=True, protocol=5)

    # 其他小对象继续用 np.save 保存
    base_info = {
        'Q': Q,
        'sin_theta_grid': sin_theta_grid,
        'theta_grid': theta_grid,
        'theta_grid_deg': theta_grid_deg,
        'M': Q.shape[0]
    }
    np.save(os.path.join(save_dir, 'base_info.npy'), base_info)

    print(f"数据已保存到 {save_dir}")
    print(f"训练集大小: {len(data_train['input'])}")
    print(f"验证集大小: {len(data_val['input'])}")


def verify_orthogonal_basis(Q):
    """验证正交基的酉性"""
    M = Q.shape[0]
    print("=== 验证正交基性质 ===")
    QH_Q = np.conj(Q).T @ Q
    identity_diff = np.linalg.norm(QH_Q - np.eye(M))
    column_norms = np.linalg.norm(Q, axis=0)
    cond_number = np.linalg.cond(Q)
    singular_values = np.linalg.svd(Q, compute_uv=False)

    print(f"1. ||Q^H Q - I|| = {identity_diff:.6e} (越小越好，<1e-5为优秀)")
    print(f"2. 列向量模长: 最小={np.min(column_norms):.6f}, 最大={np.max(column_norms):.6f} (理想值=1)")
    print(f"3. 矩阵条件数: {cond_number:.6f} (酉矩阵条件数=1)")
    print(f"4. 奇异值: 最小={np.min(singular_values):.6f}, 最大={np.max(singular_values):.6f} (理想值=1)")

    return identity_diff, column_norms, cond_number, singular_values



def load_orthobasis_data(save_dir):
    """加载正交基数据（混合加载：train/val 用 joblib，base_info 用 np.load）"""
    data_train = joblib.load(os.path.join(save_dir, 'train_data.joblib'))
    data_val = joblib.load(os.path.join(save_dir, 'val_data.joblib'))
    base_info = np.load(os.path.join(save_dir, 'base_info.npy'), allow_pickle=True).item()

    return data_train, data_val, base_info


def estimate_angles_from_coefficients_ESPRIT(
        coeff_pred, base_info, max_sources=4, wavelength=0.3
):
    """
    ESPRIT-based DOA recovery from predicted orthogonal coefficients.

    Flow:
        predicted coefficients
        -> complex coefficient vector
        -> reconstructed spatial response
        -> Hankel embedding
        -> signal subspace
        -> LS-ESPRIT
        -> spatial poles
        -> DOAs
    """

    # =========================================================
    # 1. Basic parameters
    # =========================================================
    M = base_info['M']
    Q = base_info['Q']
    d = wavelength / 2
    K = int(max_sources)

    # =========================================================
    # 2. Recover complex orthogonal coefficient vector
    # coeff_pred = [Re(c), Im(c)]
    # =========================================================
    coeff_real = coeff_pred[:M]
    coeff_imag = coeff_pred[M:]
    c_complex = coeff_real + 1j * coeff_imag

    # =========================================================
    # 3. Reconstruct spatial array response
    # y ≈ sum_k alpha_k a(theta_k)
    # =========================================================
    y = (Q @ c_complex.reshape(-1, 1)).squeeze()

    # =========================================================
    # 4. Construct Hankel data matrix
    # =========================================================
    L = max(M // 2, K + 1)
    Ncol = M - L + 1

    # Need enough rows/columns to support a rank-K subspace
    if L - 1 < K or Ncol < K:
        return np.array([]), None, None

    Y = np.zeros((L, Ncol), dtype=np.complex128)

    for i in range(L):
        Y[i, :] = y[i:i + Ncol]

    # =========================================================
    # 5. Extract rank-K signal subspace
    # Y ≈ Us Σs Vs^H
    # =========================================================
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)

    Us = U[:, :K]

    # =========================================================
    # 6. Construct two shifted SIGNAL SUBSPACES
    #
    # This is the key ESPRIT step:
    #
    # Us1 = first L-1 rows
    # Us2 = last  L-1 rows
    #
    # Us2 ≈ Us1 Psi
    # =========================================================
    Us1 = Us[:-1, :]
    Us2 = Us[1:, :]

    # =========================================================
    # 7. LS-ESPRIT rotational matrix
    #
    # Psi = Us1^† Us2
    # =========================================================
    Psi = np.linalg.pinv(Us1) @ Us2

    # =========================================================
    # 8. Eigenvalues of rotational matrix
    #
    # z_k ≈ exp(-j 2π(d/λ) sin(theta_k))
    # =========================================================
    z = np.linalg.eigvals(Psi)

    # =========================================================
    # 9. Spatial phase -> DOA
    # =========================================================
    mu = np.angle(z)

    sin_theta = -mu / (2 * np.pi * d / wavelength)

    # Numerical protection
    sin_theta = np.clip(np.real(sin_theta), -1.0, 1.0)

    theta = np.degrees(np.arcsin(sin_theta))

    # =========================================================
    # 10. Restrict to the designed angular range
    # =========================================================
    theta = np.clip(theta, -60.0, 60.0)

    # ESPRIT naturally produces K eigenvalues,
    # so do NOT use np.unique here.
    theta = np.sort(theta)

    return theta[:K], None, np.abs(z)