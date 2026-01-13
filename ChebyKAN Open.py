def save_results(Xbest, Fbest, folder="resultRe", plot_folder=None):
    """
    保存优化结果，并用 TOPSIS 方法选择最优折中解
    """
    import os
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    image_folder = plot_folder if plot_folder else os.path.join(folder, "image")
    os.makedirs(image_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    # ------------------- 绘制 Pareto 前沿 -------------------
    image_path = os.path.join(image_folder, f"TKAN_pareto_front_{timestamp}.png")
    plt.figure(figsize=(7, 5))

    # 散点
    plt.scatter(Fbest[:, 1], -Fbest[:, 0], c="g", s=30, label="Pareto Points")


    plt.xlabel("air consumption (L/min)", fontsize=12)
    plt.ylabel("airflow speed (m/s)", fontsize=12)
    plt.title("Pareto Front (MORBMO)", fontsize=12)
    plt.legend()
    # plt.grid(False, linestyle='--', alpha=0.5)
    plt.grid(False)
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Pareto 前沿图像保存成功: {image_path}")

    # ------------------- TOPSIS 选最优解 -------------------
    Y1 = -Fbest[:, 0]  # airflow speed
    Y2 = Fbest[:, 1]  # air consumption
    values = np.column_stack((Y1, Y2))

    weights = np.array([0.8, 0.2])
    indicators = np.array([1, -1])

    # 标准化
    norm_values = values / np.sqrt(np.sum(values ** 2, axis=0))
    weighted_norm = norm_values * weights

    # 理想解 & 负理想解
    ideal_best = np.max(weighted_norm * indicators, axis=0) * indicators
    ideal_worst = np.min(weighted_norm * indicators, axis=0) * indicators

    D_pos = np.linalg.norm(weighted_norm - ideal_best, axis=1)
    D_neg = np.linalg.norm(weighted_norm - ideal_worst, axis=1)

    C = D_neg / (D_pos + D_neg)  # 贴近度
    rank_idx = np.argsort(C)[::-1]

    best_idx = rank_idx[0]
    print(f"\nTOPSIS 最优折中解 index = {best_idx}, 贴近度 = {C[best_idx]:.4f}")
    print(f"最优设计变量: {Xbest[best_idx]}")
    print(f"对应目标值: F1 = {Y1[best_idx]:.4f}, F2 = {Y2[best_idx]:.4f}")

    # 绘制 TOPSIS 结果
    plt.figure(figsize=(8, 6))
    plt.scatter(Y2, Y1, alpha=0.7, label="Pareto front")
    plt.scatter(Y2[best_idx], Y1[best_idx], c="r", s=100, label="TOPSIS best")
    plt.xlabel("air consumption (L/min)", fontsize=12)
    plt.ylabel("airflow speed (m/s)", fontsize=12)
    plt.title("TOPSIS Best Solution", fontsize=12)
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(image_folder, f"topsis_{timestamp}.png"),
                dpi=300, bbox_inches="tight")
    plt.show()

    # 输出前10个
    print("\n前10个最优解:")
    for i in range(min(10, len(rank_idx))):
        idx = rank_idx[i]
        print(f"Rank {i + 1}: idx={idx}, C={C[idx]:.4f}, F1={Y1[idx]:.4f}, F2={Y2[idx]:.4f}")

    return best_idx, C


import os
import torch
import warnings
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from ChebyKAN import ChebyKANLayer
from sklearn.multioutput import MultiOutputRegressor
from data_set import remove_outliers_iqr, divide_data, scaler_data, load_data
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

plot_dir = "ChebyKAN"
os.makedirs(plot_dir, exist_ok=True)

def move_plot_files(plot_dir, filenames):
    for name in filenames:
        if os.path.exists(name):
            os.replace(name, os.path.join(plot_dir, name))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
torch.backends.cudnn.enabled = False


name = "0.3"
path = r"data3.7.xlsx"

dataset = pd.read_excel(path, sheet_name=name + "MPa")
dataset = remove_outliers_iqr(dataset)
values = dataset.values

X = values[:, :4]
Y = values[:, [7, 11]]


print("X shape:", X.shape)
print("Y shape:", Y.shape)


# 划分数据集
data_dict = divide_data(X, Y, test_size=0.3)

# 数据归一化
normalized_data, X_scaler, Y_scaler = scaler_data(data_dict)

# DataLoader
train_loader, test_loader = load_data(normalized_data, device)


class ChebyKAN(nn.Module):
    def __init__(self, layers, degree=3):
        super(ChebyKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(ChebyKANLayer(layers[i], layers[i + 1], degree))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, lmbd=1e-4):
        reg = 0.0
        for layer in self.layers:
            reg += torch.sum(torch.abs(layer.cheby_coeffs))
        return lmbd * reg



kan_model = ChebyKAN([4, 16, 2], degree=3).to(device)
optimizer = optim.AdamW(kan_model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()


from tqdm import tqdm
import torch

train_loss_values = []
val_loss_values = []
train_loss_values_target1 = []
train_loss_values_target2 = []
val_loss_values_target1 = []
val_loss_values_target2 = []

num_epochs = 300

for epoch in range(num_epochs):
    kan_model.train()
    running_loss = 0.0
    running_loss_target1 = 0.0
    running_loss_target2 = 0.0

    # === 创建 tqdm 进度条 ===
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, colour='cyan')

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = kan_model(inputs)

        # 总损失 = MSE + 正则项
        loss = criterion(outputs, targets)
        loss_target1 = torch.mean((outputs[:, 0] - targets[:, 0]) ** 2)
        loss_target2 = torch.mean((outputs[:, 1] - targets[:, 1]) ** 2)
        reg_loss = kan_model.regularization_loss(lmbd=1e-4)
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_target1 += loss_target1.item()
        running_loss_target2 += loss_target2.item()

        # 动态更新 tqdm 状态栏
        pbar.set_postfix(train_loss=running_loss / (pbar.n + 1))

    # === 计算平均训练损失 ===
    avg_train_loss = running_loss / len(train_loader)
    train_loss_values.append(avg_train_loss)
    avg_train_loss_target1 = running_loss_target1 / len(train_loader)
    avg_train_loss_target2 = running_loss_target2 / len(train_loader)
    train_loss_values_target1.append(avg_train_loss_target1)
    train_loss_values_target2.append(avg_train_loss_target2)

    # === 验证阶段 ===
    kan_model.eval()
    val_loss = 0.0
    val_loss_target1 = 0.0
    val_loss_target2 = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = kan_model(inputs)
            loss = criterion(outputs, targets)
            loss_target1 = torch.mean((outputs[:, 0] - targets[:, 0]) ** 2)
            loss_target2 = torch.mean((outputs[:, 1] - targets[:, 1]) ** 2)
            val_loss += loss.item()
            val_loss_target1 += loss_target1.item()
            val_loss_target2 += loss_target2.item()

    avg_val_loss = val_loss / len(test_loader)
    val_loss_values.append(avg_val_loss)
    avg_val_loss_target1 = val_loss_target1 / len(test_loader)
    avg_val_loss_target2 = val_loss_target2 / len(test_loader)
    val_loss_values_target1.append(avg_val_loss_target1)
    val_loss_values_target2.append(avg_val_loss_target2)


    pbar.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)
    pbar.close()


def extract_features(kan_model, data_loader, device):
    kan_model.eval()
    features, targets = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device).squeeze(1)
            labels = labels.cpu().numpy()
            for layer in kan_model.layers[:-1]:
                inputs = layer(inputs)
            features.append(inputs.cpu().numpy())
            targets.append(labels)
    return np.concatenate(features), np.concatenate(targets)

train_features, train_targets = extract_features(kan_model, train_loader, device)
test_features, test_targets = extract_features(kan_model, test_loader, device)



from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score


ensemble_size =20
predictions_ensemble = []

for seed in range(ensemble_size):
    model = MultiOutputRegressor(
        CatBoostRegressor(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            loss_function='RMSE',
            bootstrap_type='Bayesian',

            bagging_temperature=5.0,
            random_strength=3.0,
            random_seed=seed,
            allow_writing_files=False,
            verbose=0
        )
    )
    model.fit(train_features, train_targets)
    preds = model.predict(test_features)
    predictions_ensemble.append(preds)


# 计算均值与标准差（即不确定性）
predictions_ensemble = np.array(predictions_ensemble)  # shape = (ensemble_size, n_samples, n_targets)
Y_pred_mean = predictions_ensemble.mean(axis=0)



# 计算指标
mre = np.mean(np.abs((test_targets - Y_pred_mean) / test_targets))
mre_per_target = np.mean(np.abs((test_targets - Y_pred_mean) / test_targets), axis=0)

rmse = np.sqrt(mean_squared_error(test_targets, Y_pred_mean, multioutput='uniform_average'))
rmse_per_target = np.sqrt(mean_squared_error(test_targets, Y_pred_mean, multioutput='raw_values'))

r2 = r2_score(test_targets, Y_pred_mean, multioutput='uniform_average')
r2_per_target = r2_score(test_targets, Y_pred_mean, multioutput='raw_values')

print(f"CatBoost Ensemble - MRE: {mre:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
print(f"Target 1 - MRE: {mre_per_target[0]:.4f}, RMSE: {rmse_per_target[0]:.4f}, R2: {r2_per_target[0]:.4f}")
print(f"Target 2 - MRE: {mre_per_target[1]:.4f}, RMSE: {rmse_per_target[1]:.4f}, R2: {r2_per_target[1]:.4f}")


from utils.plots import plot_results_ob1, plot_results_ob2

Y_true_real = Y_scaler.inverse_transform(test_targets)
Y_pred_mean_real = Y_scaler.inverse_transform(Y_pred_mean)
plot_results_ob1(Y_pred_mean_real[:, 0], Y_true_real[:, 0],
                 ylabel_name="Airflow speed (m/s)", r2=r2_per_target[0])
plot_results_ob2(Y_pred_mean_real[:, 1], Y_true_real[:, 1],
                 ylabel_name="Air consumption (L/min)", r2=r2_per_target[1])
plot_files = [
    "Prediction 1.pdf",
    "Prediction 2.pdf"
]
move_plot_files(plot_dir, plot_files)

print("Step1 模型训练完成！")



from IMORBMO import IMORBMO as MORBMO


# 1. 预测函数
def predict_model(x):
    x = np.array(x).reshape(1, -1)
    x_norm = X_scaler.transform(x)

    kan_model.eval()
    with torch.no_grad():
        inputs = torch.tensor(x_norm, dtype=torch.float32).to(device)
        for layer in kan_model.layers[:-1]:
            inputs = layer(inputs)
        features = inputs.cpu().numpy()

    y_pred_norm = model.predict(features)
    y_pred = Y_scaler.inverse_transform(y_pred_norm)
    return y_pred[0]   # (气流速度, 耗气量)



# 2. 多目标函数
def mech_multiobj(x):
    airflow_speed, gas_consumption = predict_model(x)
    f1 = -airflow_speed      # 最大化气流速度 → 取负
    f2 = gas_consumption     # 最小化耗气量
    return np.array([f1, f2])

# 3. MORBMO 配置
nVar = 4# 输入的机械结构参数个数
numObj = 2# 目标个数

MultiObj = {
    "var_min": np.zeros(nVar),
    "var_max": np.ones(nVar) * 10,
    "fun": mech_multiobj,
    "numOfObj": numObj,
    "nVar": nVar,
    "name": "MechOpt"
}

params = {
    "Np": 100,# 种群数量
    "Nr": 25,# 参考点数量
    "maxgen": 50# 最大迭代次数
}


# 4. 运行优化
Xbest, Fbest = MORBMO(params, MultiObj)
# 结果展示
print("最终解集 Xbest 的形状:", Xbest.shape)
print("最终帕累托前沿 Fbest 的形状:", Fbest.shape)
# 还原真实速度
real_Fbest = np.copy(Fbest)
real_Fbest[:, 0] = -real_Fbest[:, 0]
print("Pareto前沿前5个解 (气流速度, 耗气量):")
print(real_Fbest[:5])
# 保存结果 & TOPSIS
best_idx, C = save_results(Xbest, Fbest, folder=plot_dir, plot_folder=plot_dir)








