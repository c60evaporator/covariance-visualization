# %% 独立な2変数の正規分布
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# パラメータ
MU1 = 1
MU2 = 1
SIGMA1 = 2
SIGMA2 = 1
# (x1,x2)格子点を作成
x1_grid = np.linspace(-6, 6, 100)
x2_grid = np.linspace(-6, 6, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# 正規分布の同時分布関数
def norm_2d(x1, x2):
    p1 = np.exp(-np.square(x1-MU1)/(2*SIGMA1**2))/np.sqrt(2*np.pi*SIGMA1)
    p2 = np.exp(-np.square(x2-MU2)/(2*SIGMA2**2))/np.sqrt(2*np.pi*SIGMA2)
    p = p1*p2
    return p

# 同時分布をプロットするメソッド
def plot_joint_distribution(x1, x2, p):
    # 曲面をプロット
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x1", size = 16)  # x1軸
    ax.set_ylabel("x2", size = 16)  # x2軸
    ax.set_zlabel("p(x1,x2)", size = 16)  # p軸
    ax.plot_surface(x1, x2, p, cmap = "YlGn_r")
    plt.show()
    # 等高線をプロット
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    plt.contour(x1, x2, p, cmap="YlGn_r")
    ax.set_xlabel("x1", size = 16)  # x1軸
    ax.set_ylabel("x2", size = 16)  # x2軸
    plt.grid()
    plt.show()

# 同時分布を計算
P = norm_2d(X1, X2)
# 同時分布プロット実行
plot_joint_distribution(X1, X2, P)

# %% 標準正規分布を拡大→回転→平行移動
# 2次元行列とベクトルの積を計算する関数(@では配列に一括実行できないため自作)
def mat_vec_multiplication(matrix, vec1, vec2):
    dst1 = matrix[0,0]*vec1 + matrix[0,1]*vec2
    dst2 = matrix[1,0]*vec1 + matrix[1,1]*vec2
    return dst1, dst2

# 変換前の同時分布関数（標準正規分布）
def norm_std_2d(y1, y2):
    p1 = np.exp(-np.square(y1))/np.sqrt(2*np.pi)
    p2 = np.exp(-np.square(y2))/np.sqrt(2*np.pi)
    p = p1*p2
    return p

###### 座標変換前 ######
P = norm_std_2d(X1, X2)  # 変換前の同時分布計算
plot_joint_distribution(X1, X2, P)  # 同時分布をプロット

###### スキュー ######8
PHI = 45  # スキューの角度
phi_rad = PHI*2*np.pi/360
skew_mat = np.array([[1, np.tan(phi_rad)],
                     [0, 1]])  # スキューの行列
mag_skew_inv = np.linalg.inv(skew_mat)  # スキューの逆行列
y1, y2 = mat_vec_multiplication(mag_skew_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### 拡大縮小 ######
mag_mat = np.array([[SIGMA1,0],
                    [0,SIGMA2]])  # 拡大縮小行列
mag_inv = np.linalg.inv(mag_mat@skew_mat)  # スキュー＋拡大縮小の逆行列
y1, y2 = mat_vec_multiplication(mag_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### 反転 ######
mir_mat = np.array([[1,0],
                    [0,-1]])  # 拡大縮小行列
mir_inv = np.linalg.inv(mir_mat@mag_mat@skew_mat)  # スキュー＋拡大縮小＋反転の逆行列
y1, y2 = mat_vec_multiplication(mir_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### 回転 ######
THETA = 60  # 回転角度
theta_rad = THETA*2*np.pi/360  # ラジアンに変換
rot_mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                    [np.sin(theta_rad), np.cos(theta_rad)]])  # 回転行列
rot_inv = np.linalg.inv(rot_mat@mir_mat@mag_mat@skew_mat)  # スキュー＋拡大縮小＋反転＋回転の逆行列
y1, y2 = mat_vec_multiplication(rot_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### 平行移動 ######
# 平行移動ベクトル
y1, y2 = X1-MU1, X2-MU2  # 平行移動の逆変換
y1, y2 = mat_vec_multiplication(rot_inv, y1, y2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

# %% 特異値分解
A = rot_mat@mir_mat@mag_mat@skew_mat  # スキュー＋拡大縮小＋反転＋回転の行列
print(A)
# Aを特異値分解
U, S, Vt = np.linalg.svd(A, full_matrices=True)
a = U[1,1]
# Uの反転有無を判定 (対角成分の符号が一致していれば反転なし)
if U[0,0] * U[1,1] > 0:
    U_rotmir = [U]
else:
    U_rotmir = [np.array([[U[0,0], -U[0,1]],[-U[1,0], -U[1,1]]]), np.array([[1,0],[0,-1]])]
# Vの反転有無を判定 (対角成分の符号が一致していれば反転なし)
if Vt[0,0] * Vt[1,1] > 0:
    V_rotmir = [Vt]
else:
    V_rotmir = [np.array([[Vt[0,0], -Vt[0,1]],[-Vt[1,0], -Vt[1,1]]]), np.array([[1,0],[0,-1]])]

###### 座標変換前 ######
P = norm_std_2d(X1, X2)  # 変換前の同時分布計算
plot_joint_distribution(X1, X2, P)  # 同時分布をプロット

###### Vによる回転 ######
v_rot_inv = np.linalg.inv(V_rotmir[0])  # V回転の逆行列
y1, y2 = mat_vec_multiplication(v_rot_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### Vによる反転(反転存在時のみ) ######
if len(V_rotmir) == 2:
    v_mir_inv = np.linalg.inv(V_rotmir[1]@V_rotmir[0])  # V回転+V反転の逆行列
    y1, y2 = mat_vec_multiplication(v_mir_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
    P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
    plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### 拡大縮小 ######
S_mag_mat = np.array([[S[0],0],[0,S[1]]])
mag_inv = np.linalg.inv(S_mag_mat@Vt)  # V＋拡大縮小の逆行列
y1, y2 = mat_vec_multiplication(mag_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### Uによる反転(反転存在時のみ) ######
if len(U_rotmir) == 2:
    u_mir_inv = np.linalg.inv(U_rotmir[1]@S_mag_mat@Vt)  # V＋拡大縮小+U反転の逆行列
    y1, y2 = mat_vec_multiplication(u_mir_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
    P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
    plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### Uによる回転 ######
u_rot_inv = np.linalg.inv(U@S_mag_mat@Vt)  # V＋拡大縮小+Uの逆行列
y1, y2 = mat_vec_multiplication(u_rot_inv, X1, X2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

###### 平行移動 ######
# 平行移動ベクトル
y1, y2 = X1-MU1, X2-MU2  # 平行移動の逆変換
y1, y2 = mat_vec_multiplication(u_rot_inv, y1, y2)  # 変換後の(X1, X2)が変換前の(Y1, Y2)座標のどこに相当するか計算
P = norm_std_2d(y1, y2)  # 変換前の同時分布関数（標準正規分布）
plot_joint_distribution(X1, X2, P)  # 変換後の(X1, X2)座標で同時分布をプロット

# %% 非独立な2変数の正規分布の断面
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MU1 = 1  # x1の平均
MU2 = 1  # x2の平均
SIGMA1 = 2  # y1方向の分散
SIGMA2 = 1  # y2方向の分散
THETA = 30  # 回転角度
theta_rad = THETA*2*np.pi/360  # ラジアンに変換

# 回転行列
rot_mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],[np.sin(theta_rad), np.cos(theta_rad)]])
# 拡大縮小
mag_mat = np.array([[SIGMA1,0],[0,SIGMA2]])
inv = np.linalg.inv(rot_mat@mag_mat)


# 同時分布関数
def p(x1, x2):
    y1, y2 = mat_vec_multiplication(inv, x1, x2)
    p1 = np.exp(-np.square(y1))/np.sqrt(2*np.pi)
    p2 = np.exp(-np.square(y2))/np.sqrt(2*np.pi)
    p = p1*p2
    return p

# (x1,x2)格子点を作成
x1 = np.linspace(-6, 6, 100)
x2 = np.linspace(-6, 6, 100)
X1, X2 = np.meshgrid(x1, x2)
# 同時分布を計算
P = p(X1, X2)

# 曲面をプロット
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("x1", size = 16)  # x1軸
ax.set_ylabel("x2", size = 16)  # x2軸
ax.set_zlabel("p(x1,x2)", size = 16)  # p軸
ax.plot_surface(X1, X2, P, cmap = "YlGn_r")
plt.show()
# 等高線をプロット
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
plt.contour(X1, X2, P, cmap="YlGn_r")
ax.set_xlabel("x1", size = 16)  # x1軸
ax.set_ylabel("x2", size = 16)  # x2軸


# %%
