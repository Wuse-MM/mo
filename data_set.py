import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.func import evaluate_forecasts
import pandas as pd
import numpy as np


# 使用IQR方法剔除异常值
def remove_outliers_iqr(df):
    # 计算Q1, Q3和IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # 设置阈值为1.5倍IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 筛选出没有异常值的行
    df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

    return df_no_outliers


def get_all(data_dict):
    X_train = data_dict["X_train"]
    Y_train = data_dict["Y_train"]
    X_test = data_dict["X_test"]
    Y_test = data_dict["Y_test"]

    return X_train, X_test, Y_train, Y_test


def divide_data(X, Y, test_size=0.1, val_size=0.1):  # 完成数据的划分

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=test_size, random_state=42, shuffle=True)
    data_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
    }

    # 训练集，验证集，测试集
    return data_dict


def scaler_data(data_dict):
    # 创建新的字典作为副本
    normalized_data = data_dict.copy()

    # 创建 MinMaxScaler 实例
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    # 对输入数据（X）进行归一化
    normalized_data["X_train"] = X_scaler.fit_transform(normalized_data["X_train"])
    normalized_data["X_test"] = X_scaler.transform(normalized_data["X_test"])
    print(normalized_data["Y_train"].shape)
    shape = normalized_data["Y_train"].shape
    if len(shape) == 2:
        normalized_data["Y_train"] = Y_scaler.fit_transform(normalized_data["Y_train"])
        normalized_data["Y_test"] = Y_scaler.transform(normalized_data["Y_test"])
    else:
        normalized_data["Y_train"] = Y_scaler.fit_transform(normalized_data["Y_train"].reshape(-1, 1))
        normalized_data["Y_test"] = Y_scaler.transform(normalized_data["Y_test"].reshape(-1, 1))

    # 对输出数据（Y）进行归一化


    return normalized_data, X_scaler, Y_scaler


def load_data(normalized_data, device, batch_size=32):
    X_train, X_test, Y_train, Y_test = get_all(normalized_data)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    # 转换为Tensor并移动到设备
    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).unsqueeze(-1).to(device)  # 增加通道维度和高度维度
    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # 增加通道维度和高度维度
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)  # 增加通道维度和高度维度
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)

    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).unsqueeze(-1).to(device)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 遍历 test_loader 提取归一化后的数据
def restore_original_data(test_loader, X_scaler, Y_scaler, device):
    all_X_normalized = []
    all_Y_normalized = []

    # 提取批次数据
    for X_batch, Y_batch in test_loader:
        all_X_normalized.append(X_batch)
        all_Y_normalized.append(Y_batch)

    # 合并所有批次数据
    all_X_normalized = torch.cat(all_X_normalized, dim=0).squeeze(-1).squeeze(1).cpu().numpy()  # 转为 NumPy 并移除多余维度
    all_Y_normalized = torch.cat(all_Y_normalized, dim=0).cpu().numpy()

    # 使用 MinMaxScaler 还原数据
    X_original = X_scaler.inverse_transform(all_X_normalized)
    Y_original = Y_scaler.inverse_transform(all_Y_normalized)

    return X_original, Y_original


def get_metrics(model, features, lables):
    predicted_data = model.predict(features).reshape(-1, 1)
    metrics = evaluate_forecasts(lables, predicted_data)
    return metrics, predicted_data
