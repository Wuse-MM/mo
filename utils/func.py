import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
import torch
import joblib
from pyswarm import pso

def print_cnn_params(model):
    total_params = 0
    print("===== CNN 模型参数与维度 =====")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"参数名称: {name}, 维度: {param.shape}, 参数量: {param.numel()}")
            total_params += param.numel()
    print(f"CNN 模型总参数量: {total_params}")
def print_rf_params(model):
    print("\n===== RandomForest 模型参数 =====")
    params = model.get_params()
    for key, value in params.items():
        print(f"{key}: {value}")

def optimize_cnn_features(cnn_model, X_scaler, Y_scaler, device, n=20, Tmax=50, LB=None, UB=None):
    # 默认的 PSO 参数
    if LB is None:
        LB = [0] * 4  # 下界
    if UB is None:
        UB = [1] * 4  # 上界

    def objective_function_HIDMSPSO(X_features_norm):
        """
        目标函数，用于 PSO 优化，通过 CNN 模型进行预测。
        """
        X_features = np.array(X_features_norm).reshape(1, 1, 4, 1)

        X_tensor = torch.tensor(X_features).float().to(device)
        with torch.no_grad():
            # 使用 CNN 模型进行预测
            cnn_predictions = cnn_model(X_tensor)

        # 进行逆归一化
        cnn_predictions_original = Y_scaler.inverse_transform(cnn_predictions.cpu().detach().numpy().reshape(-1, 1))[0][
            0]

        return cnn_predictions_original

    # 使用 PSO 优化
    Best_Features, Best_Prediction = pso(
        objective_function_HIDMSPSO,
        LB, UB,
        swarmsize=n,
        maxiter=Tmax
    )

    # 逆归一化特征
    Best_Features_original = X_scaler.inverse_transform(np.array(Best_Features).reshape(1, -1))

    return Best_Features_original, Best_Prediction


def optimize_and_predict(cnn_model, skl_model, X_scaler, Y_scaler, device, n=20, Tmax=50, LB=None, UB=None, d=4):
    if LB is None:
        LB = [0] * d  # 默认下界
    if UB is None:
        UB = [1] * d  # 默认上界

    def objective_function_HIDMSPSO(X_features_norm):
        X_features = np.array(X_features_norm).reshape(1, 1, 4, 1)  # 调整为 CNN 所需的输入形状

        X_tensor = torch.tensor(X_features).float().to(device)
        with torch.no_grad():
            # 使用 CNN 模型提取特征
            cnn_features = cnn_model(X_tensor)

        # 使用 SVR 模型进行预测
        svr_predictions = skl_model.predict(cnn_features.cpu().detach().numpy())  # 从 GPU 移回 CPU 进行预测

        # 进行逆归一化
        svr_predictions_original = Y_scaler.inverse_transform(svr_predictions.reshape(-1, 1))[0][0]

        return svr_predictions_original

    # 使用 PSO 优化
    Best_Features, Best_Prediction = pso(
        lambda X: objective_function_HIDMSPSO(X),
        LB, UB, swarmsize=n, maxiter=Tmax
    )

    # 反归一化最优特征
    Best_Features_original = X_scaler.inverse_transform(np.array(Best_Features).reshape(1, -1))
    return Best_Features_original, Best_Prediction


def save_models(cnn_model, ml_model, X_scaler, Y_scaler, save_dir="saved_models"):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存 CNN 模型
    cnn_model_path = os.path.join(save_dir, "best_cnn_model.pth")
    torch.save(cnn_model.state_dict(), cnn_model_path)
    print(f"✅ CNN 模型已保存至 {cnn_model_path}")

    # 保存 机器学习模型
    ml_model_path = os.path.join(save_dir, "best_ml_model.pkl")
    joblib.dump(ml_model, ml_model_path)
    print(f"✅ 机器学习模型已保存至 {ml_model_path}")

    # 保存 标准化器
    scaler_path = os.path.join(save_dir, "scalers.pkl")
    joblib.dump((X_scaler, Y_scaler), scaler_path)
    print(f"✅ 数据标准化器已保存至 {scaler_path}")


def add_noise(predicted_data, noise_level=0.005):
    # noise_level 控制扰动的大小
    noise = np.random.normal(0, noise_level, predicted_data.shape)  # 生成正态分布的噪声
    return predicted_data + noise


def judge_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"路径 '{path}' 不存在，已创建该路径。")


def write_to_excel(true_data, predicted_data, path, name="", Xtest=""):
    # 检查路径是否存在，如果不存在则创建
    judge_path_exists(path)

    # 创建结果 DataFrame
    if Xtest != "":
        results_df = pd.DataFrame(Xtest, columns=[f'Feature_{i}' for i in range(Xtest.shape[1])])
    else:
        results_df = pd.DataFrame()

    results_df['True'] = true_data
    results_df['Predicted'] = predicted_data

    # 获取文件名
    if name == "":
        name = input("请输入excel名字：")

    # 拼接文件路径
    file_path = os.path.join(path, f"{name}.xlsx")

    # 写入 Excel 文件
    results_df.to_excel(file_path, index=False)
    print(f"Excel 写入完成，文件保存至: {file_path}")


def save_results_to_txt(metrics, min_gas_predicted, corresponding_features, filename='evaluation_results.txt'):
    with open(filename, 'a', encoding="utf-8") as f:  # 使用 'a' 模式进行追加写入
        f.write("\n=== New Evaluation Results ===\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")  # 换行分隔不同模型或运行的结果
        f.write(f"Predicted minimum gas consumption: {min_gas_predicted:.4f} L/min\n")
        f.write(f"Corresponding features for minimum gas consumption: {corresponding_features}\n")


# 计算评估指标
def mape(y_true, y_pred):
    non_zero_indices = y_true != 0
    if np.sum(non_zero_indices) == 0:
        return float('inf')
    return np.mean(np.abs((y_pred[non_zero_indices] - y_true[non_zero_indices]) / y_true[non_zero_indices])) * 100


# 定义 IA 指数的计算函数
def index_of_agreement(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_true - mean_y_true) + np.abs(y_pred - mean_y_true)) ** 2)
    return 1 - (numerator / denominator)


# 定义 TIC 指数的计算函数
def theils_inequality_coefficient(y_true, y_pred):
    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    return numerator / denominator


def evaluate_forecasts(true_data, predicted_data):
    mse = mean_squared_error(true_data, predicted_data)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_data, predicted_data)
    # mape_value = mape(true_data, predicted_data)  # 调用 mape 函数
    # tic = theils_inequality_coefficient(true_data, predicted_data)
    # ia = index_of_agreement(true_data, predicted_data)
    r2 = r2_score(true_data, predicted_data)

    # return mse, rmse, mae, mape_value, tic, ia, r2
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        # 'TIC': tic,
        # 'IA': ia,
        # 'MAPE': mape_value
    }
