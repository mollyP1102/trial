import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
import matplotlib as mpl

'''
数据生成：生成基于正弦函数的模拟数据，添加高斯噪声，用于训练和测试。
RBF核函数：实现径向基函数（RBF）核，用于计算训练和测试数据之间的协方差。
高斯过程回归：实现GPR模型，计算后验均值和协方差，支持先验样本生成。
超参数优化：通过最小化负对数边际似然（Negative Log Marginal Likelihood）优化RBF核的超参数（长度尺度、信号标准差、噪声标准差）。
可视化：展示优化前后的预测结果（后验均值和置信区间）以及先验分布的对比。
'''
# 设置字体以支持中文显示，避免中文乱码
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示
except:
    print("中文字体设置失败，将使用英文标签")

# 设置全局图形参数
plt.rcParams['figure.figsize'] = [10, 6]  # 设置图形尺寸
plt.rcParams['font.size'] = 12  # 设置字体大小

# 定义径向基函数（RBF）核
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    计算RBF核矩阵（平方指数核）
    
    参数:
        X1: 输入数组 (m x d)，第一组数据点
        X2: 输入数组 (n x d)，第二组数据点
        length_scale: 长度尺度参数，控制函数平滑度
        sigma_f: 信号标准差，控制函数幅度
    
    返回:
        核矩阵 (m x n)，表示点之间的协方差
    """
    sq_dist = cdist(X1, X2, 'sqeuclidean')  # 计算点对之间的平方欧几里得距离
    return sigma_f**2 * np.exp(-0.5 * sq_dist / length_scale**2)  # RBF核公式

# 设置随机种子，确保结果可重复
np.random.seed(42)
n_train = 10  # 训练数据点数量
n_test = 100  # 测试数据点数量

# 生成训练数据
X_train = np.random.uniform(-5, 5, n_train).reshape(-1, 1)  # 在[-5, 5]均匀分布的输入
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(n_train)  # 正弦函数加高斯噪声

# 生成测试数据
X_test = np.linspace(-7, 7, n_test).reshape(-1, 1)  # 在[-7, 7]均匀分布的测试点

# 定义高斯过程回归函数
def gp_regression(X_train, y_train, X_test, kernel, sigma_n=0.1, return_prior=False):
    """
    实现高斯过程回归，计算后验均值和协方差
    
    参数:
        X_train: 训练输入 (n x d)
        y_train: 训练输出 (n,)
        X_test: 测试输入 (m x d)
        kernel: 核函数，计算协方差
        sigma_n: 观测噪声标准差
        return_prior: 是否返回先验样本
    
    返回:
        mu: 后验均值 (m,)
        cov: 后验协方差矩阵 (m x m)
        prior_samples: 先验样本 (m x 5)，若return_prior=True
    """
    # 计算核矩阵：训练-训练、训练-测试、测试-测试
    K = kernel(X_train, X_train)  # 训练集协方差矩阵
    K_star = kernel(X_train, X_test)  # 训练-测试协方差矩阵
    K_star_star = kernel(X_test, X_test)  # 测试集协方差矩阵
    
    # 添加观测噪声到训练集核矩阵的对角线
    K += sigma_n**2 * np.eye(len(X_train))
    
    # Cholesky分解：K = L @ L.T
    try:
        L = cholesky(K, lower=True)  # 下三角分解
    except:
        K += 1e-6 * np.eye(len(X_train))  # 添加小正则化项以确保正定
        L = cholesky(K, lower=True)
    
    # 解线性系统 L @ alpha = y_train
    alpha = solve_triangular(L, y_train, lower=True)
    
    # 计算后验均值：mu = K_star^T @ (K^-1 @ y_train)
    mu = K_star.T @ solve_triangular(L.T, alpha, lower=False)
    
    # 计算后验协方差：cov = K_star_star - K_star^T @ K^-1 @ K_star
    v = solve_triangular(L, K_star, lower=True)
    cov = K_star_star - v.T @ v
    
    if return_prior:
        # 生成先验样本
        prior_mean = np.zeros(len(X_test))  # 先验均值为0
        prior_cov = K_star_star  # 先验协方差为测试集核矩阵
        try:
            L_prior = cholesky(prior_cov + 1e-6 * np.eye(len(prior_cov)), lower=True)
        except:
            L_prior = cholesky(prior_cov + 1e-3 * np.eye(len(prior_cov)), lower=True)
        prior_samples = prior_mean.reshape(-1, 1) + L_prior @ np.random.randn(len(X_test), 5)
        return mu, cov, prior_samples
    
    return mu, cov

# 定义负对数边际似然函数，用于超参数优化
def neg_log_marginal_likelihood(params, X, y):
    """
    计算负对数边际似然，用于超参数优化
    
    参数:
        params: 超参数向量 [length_scale, sigma_f, sigma_n]
        X: 训练输入数据 (n x d)
        y: 训练输出数据 (n,)
    
    返回:
        负对数边际似然值（最小化目标）
    """
    length_scale, sigma_f, sigma_n = params
    
    # 定义当前超参数的RBF核
    kernel = lambda X1, X2: rbf_kernel(X1, X2, length_scale, sigma_f)
    
    n = len(y)
    K = kernel(X, X) + sigma_n**2 * np.eye(n)  # 训练集核矩阵加噪声
    
    try:
        L = cholesky(K, lower=True)  # Cholesky分解
    except:
        K += 1e-6 * np.eye(n)  # 添加正则化项
        L = cholesky(K, lower=True)
    
    alpha = solve_triangular(L, y, lower=True)
    
    # 对数边际似然公式：-0.5 * (y^T K^-1 y + log|K| + n*log(2π))
    data_fit = -0.5 * np.dot(alpha, alpha)  # 数据拟合项
    complexity = -np.sum(np.log(np.diag(L)))  # 复杂度惩罚项
    constant = -0.5 * n * np.log(2 * np.pi)  # 常数项
    
    log_likelihood = data_fit + complexity + constant
    return -log_likelihood  # 返回负值以进行最小化

# 初始超参数猜测
initial_params = [1.0, 1.0, 0.1]  # [length_scale, sigma_f, sigma_n]

# 设置超参数边界
bounds = [(1e-3, 10.0), (1e-3, 10.0), (1e-3, 1.0)]  # 限制参数范围

# 使用L-BFGS-B算法优化超参数
result = minimize(
    neg_log_marginal_likelihood, 
    initial_params, 
    args=(X_train, y_train),
    bounds=bounds,
    method='L-BFGS-B'
)

# 提取优化后的超参数
opt_length_scale, opt_sigma_f, opt_sigma_n = result.x
print(f"优化后的超参数: length_scale={opt_length_scale:.3f}, sigma_f={opt_sigma_f:.3f}, sigma_n={opt_sigma_n:.3f}")
print(f"优化后的负对数边际似然值: {result.fun:.3f}")

# 定义优化后的核函数
opt_kernel = lambda X1, X2: rbf_kernel(X1, X2, opt_length_scale, opt_sigma_f)

# 使用优化后的超参数进行GP回归
mu_opt, cov_opt, prior_samples_opt = gp_regression(
    X_train, y_train, X_test, opt_kernel, opt_sigma_n, return_prior=True
)

# 计算95%置信区间
std_opt = np.sqrt(np.diag(cov_opt))  # 后验标准差
upper_opt = mu_opt + 1.96 * std_opt  # 上界
lower_opt = mu_opt - 1.96 * std_opt  # 下界

# 创建子图，比较优化前后的回归结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 使用初始超参数进行回归
length_scale_init = 1.0
sigma_f_init = 1.0
sigma_n_init = 0.1
kernel_init = lambda X1, X2: rbf_kernel(X1, X2, length_scale_init, sigma_f_init)
mu_init, cov_init = gp_regression(X_train, y_train, X_test, kernel_init, sigma_n_init)
std_init = np.sqrt(np.diag(cov_init))
upper_init = mu_init + 1.96 * std_init
lower_init = mu_init - 1.96 * std_init

# 绘制初始超参数结果
ax1.plot(X_train, y_train, 'ro', markersize=8, label='Training Data')
ax1.plot(X_test, mu_init, 'b-', lw=2, label='Posterior Mean')
ax1.fill_between(X_test.ravel(), lower_init, upper_init, alpha=0.3, color='blue', label='95% Confidence')
ax1.plot(X_test, np.sin(X_test), 'g--', lw=2, label='True Function')
ax1.set_title(f'Initial Hyperparameters\n(length_scale={length_scale_init}, sigma_f={sigma_f_init}, sigma_n={sigma_n_init})')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-7, 7)

# 绘制优化后超参数结果
ax2.plot(X_train, y_train, 'ro', markersize=8, label='Training Data')
ax2.plot(X_test, mu_opt, 'b-', lw=2, label='Posterior Mean')
ax2.fill_between(X_test.ravel(), lower_opt, upper_opt, alpha=0.3, color='blue', label='95% Confidence')
ax2.plot(X_test, np.sin(X_test), 'g--', lw=2, label='True Function')
ax2.set_title(f'Optimized Hyperparameters\n(length_scale={opt_length_scale:.3f}, sigma_f={opt_sigma_f:.3f}, sigma_n={opt_sigma_n:.3f})')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-7, 7)

plt.tight_layout()
plt.savefig('gp_optimization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建子图，比较优化前后的先验分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 初始先验分布
kernel_init = lambda X1, X2: rbf_kernel(X1, X2, length_scale_init, sigma_f_init)
prior_cov_init = kernel_init(X_test, X_test)
L_prior_init = cholesky(prior_cov_init + 1e-6 * np.eye(len(prior_cov_init)), lower=True)
prior_samples_init = L_prior_init @ np.random.randn(len(X_test), 5)

ax1.plot(X_test, np.zeros_like(X_test), 'k--', lw=2, label='Prior Mean')
for i in range(5):
    ax1.plot(X_test, prior_samples_init[:, i], lw=1)
ax1.set_title(f'Initial Prior\n(length_scale={length_scale_init}, sigma_f={sigma_f_init})')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-7, 7)

# 优化后的先验分布
prior_cov_opt = opt_kernel(X_test, X_test)
L_prior_opt = cholesky(prior_cov_opt + 1e-6 * np.eye(len(prior_cov_opt)), lower=True)
prior_samples_opt = L_prior_opt @ np.random.randn(len(X_test), 5)

ax2.plot(X_test, np.zeros_like(X_test), 'k--', lw=2, label='Prior Mean')
for i in range(5):
    ax2.plot(X_test, prior_samples_opt[:, i], lw=1)
ax2.set_title(f'Optimized Prior\n(length_scale={opt_length_scale:.3f}, sigma_f={opt_sigma_f:.3f})')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-7, 7)

plt.tight_layout()
plt.savefig('gp_prior_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 在现有代码的最后添加以下内容

# 生成一个全新的测试集（与之前的X_test不同）
np.random.seed(123)  # 设置不同的随机种子
X_test_new = np.random.uniform(-6, 6, 20).reshape(-1, 1)  # 新的测试点，20个随机点
y_test_true_new = np.sin(X_test_new).ravel()  # 真实值（无噪声）

print(f"新测试集形状: {X_test_new.shape}")
print(f"新测试集范围: [{X_test_new.min():.2f}, {X_test_new.max():.2f}]")

# 使用优化后的超参数对新测试集进行预测
mu_test_new, cov_test_new = gp_regression(X_train, y_train, X_test_new, opt_kernel, opt_sigma_n)

# 计算预测的标准差和置信区间
std_test_new = np.sqrt(np.diag(cov_test_new))
upper_test_new = mu_test_new + 1.96 * std_test_new
lower_test_new = mu_test_new - 1.96 * std_test_new

# 计算预测性能指标
mse = np.mean((mu_test_new - y_test_true_new) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(mu_test_new - y_test_true_new))

print(f"\n新测试集预测性能:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 创建新测试集的预测可视化
plt.figure(figsize=(14, 10))

# 首先绘制完整的函数范围（用于背景参考）
X_full = np.linspace(-7, 7, 300).reshape(-1, 1)
y_full = np.sin(X_full).ravel()
plt.plot(X_full, y_full, 'g-', lw=2, alpha=0.7, label='True Function (sin(x))')

# 绘制训练数据
plt.plot(X_train, y_train, 'ro', markersize=10, label='Training Data', zorder=10)

# 绘制新测试集的预测结果
plt.plot(X_test_new, mu_test_new, 'bo', markersize=8, label='Predicted Mean (Test Points)', zorder=9)
plt.errorbar(X_test_new.ravel(), mu_test_new, yerr=1.96*std_test_new, 
             fmt='none', ecolor='blue', elinewidth=2, capsize=4, capthick=2,
             label='95% Confidence Interval', alpha=0.7)

# 连接预测点以便更好地观察趋势
sort_idx = np.argsort(X_test_new.ravel())
plt.plot(X_test_new[sort_idx], mu_test_new[sort_idx], 'b--', lw=1, alpha=0.5, label='Prediction Trend')

# 添加性能指标文本
textstr = '\n'.join((
    f'新测试集性能:',
    f'RMSE: {rmse:.4f}',
    f'MAE: {mae:.4f}',
    f'测试点数: {len(X_test_new)}',
    f'优化超参数:',
    f'length_scale: {opt_length_scale:.3f}',
    f'sigma_f: {opt_sigma_f:.3f}',
    f'sigma_n: {opt_sigma_n:.3f}'))
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('高斯过程回归 - 新测试集预测结果\n（基于训练数据和优化后的核函数）')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim(-7, 7)
plt.ylim(-1.8, 1.8)

plt.tight_layout()
plt.savefig('gp_new_test_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# 额外：显示预测的统计信息
print(f"\n预测统计信息:")
print(f"预测均值范围: [{mu_test_new.min():.3f}, {mu_test_new.max():.3f}]")
print(f"预测标准差范围: [{std_test_new.min():.3f}, {std_test_new.max():.3f}]")
print(f"平均不确定性: {std_test_new.mean():.3f}")

# 检查哪些预测点的置信区间包含真实值
within_interval = np.logical_and(y_test_true_new >= lower_test_new, 
                                y_test_true_new <= upper_test_new)
coverage_rate = np.mean(within_interval) * 100
print(f"95%置信区间覆盖率: {coverage_rate:.1f}%")

# 可视化预测误差分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
errors = mu_test_new - y_test_true_new
plt.scatter(X_test_new, errors, c='red', s=50, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.fill_between(X_test_new.ravel(), -1.96*std_test_new, 1.96*std_test_new, 
                alpha=0.2, color='gray', label='±1.96σ')
plt.xlabel('x')
plt.ylabel('预测误差')
plt.title('预测误差 vs x坐标')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(errors, bins=10, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='零误差')
plt.xlabel('预测误差')
plt.ylabel('频次')
plt.title('预测误差分布')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('gp_prediction_error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("新测试集预测完成！")