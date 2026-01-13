import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 定义字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def plot_regression_curve_plots(y_train, pred_train, y_test, pred_test, file_name=""):
    # 保证输入为一维
    y_train = np.array(y_train).ravel()
    pred_train = np.array(pred_train).ravel()
    y_test = np.array(y_test).ravel()
    pred_test = np.array(pred_test).ravel()

    # 自定义颜色映射
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # 设置误差范围
    error_range = 2  # 可调整误差范围

    # 计算图像范围
    all_values = np.concatenate([y_train, pred_train, y_test, pred_test])
    x_axis_start = np.min(all_values) * 0.95
    x_axis_end = np.max(all_values) * 1.05

    # 计算绝对误差
    train_abs_error = np.abs(y_train - pred_train)
    test_abs_error = np.abs(y_test - pred_test)

    # 组合误差以标准化颜色映射
    combined_abs_error = np.concatenate([train_abs_error, test_abs_error])
    norm = plt.Normalize(vmin=combined_abs_error.min(), vmax=combined_abs_error.max())

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制训练集散点图
    sc_train = ax.scatter(
        y_train, pred_train, c=train_abs_error, cmap=cmap, norm=norm, alpha=0.8,
        marker='o', edgecolor='k', label='Training Set'
    )

    # 绘制测试集散点图
    sc_test = ax.scatter(
        y_test, pred_test, c=test_abs_error, cmap=cmap, norm=norm, alpha=0.8,
        marker='v', edgecolor='k', label='Test Set'
    )

    # 设置坐标轴范围
    ax.set_xlim(x_axis_start, x_axis_end)
    ax.set_ylim(x_axis_start, x_axis_end)

    # 添加参考线和误差虚线
    ax.plot([x_axis_start, x_axis_end], [x_axis_start, x_axis_end], 'k-', label='Ideal Fit')
    ax.plot([x_axis_start, x_axis_end], [x_axis_start + error_range, x_axis_end + error_range], 'k--')
    ax.plot([x_axis_start, x_axis_end], [x_axis_start - error_range, x_axis_end - error_range], 'k--')

    # 图形标题和标签
    ax.set_title('Predicted vs True Values with Error Ranges (±{})'.format(error_range))
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.legend(loc='upper left', frameon=True)

    # 添加颜色条
    cbar = fig.colorbar(sc_test, ax=ax, shrink=0.7, location='right')
    cbar.set_label('Absolute Error')

    # 显示网格和图像
    ax.grid(False)
    plt.tight_layout()

    # 保存图像（如果提供了文件名）
    # if file_name != "":
    #     plt.savefig(f'N:\\tf_torch\\pythonProject\\All_CNN_Attention\\Figure/{file_name}.png')

    plt.show()


def plot_regression_curve_plots1(y_train, pred_train, y_test, pred_test, file_name=""):
    plt.figure(figsize=(8, 6))
    # 定义自定义颜色映射，从红 -> 橙 -> 黄 -> 绿 -> 青 -> 蓝 -> 紫
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # 设置误差范围变量
    error_range = 3  # 您可以随时修改此值，影响误差范围

    # 计算训练和测试的最大最小值
    all_values = np.concatenate([y_train, pred_train, y_test, pred_test])
    x_axis_start = np.min(all_values) * 0.95  # 设定一个稍微低于最小值的范围
    x_axis_end = np.max(all_values) * 1.05  # 设定一个稍微高于最大值的范围
    y_axis_start = x_axis_start  # 与x轴范围相同
    y_axis_end = x_axis_end  # 与x轴范围相同

    # 计算训练集和测试集的绝对误差
    train_abs_error = np.abs(y_train - pred_train.ravel())
    test_abs_error = np.abs(y_test - pred_test.ravel())

    combined_max = max(train_abs_error.max(), test_abs_error.max())
    combined_min = min(train_abs_error.min(), test_abs_error.min())
    print(combined_max, combined_min)

    # 组合训练集和测试集的真实值、预测值和绝对误差
    # combined_abs_error = np.concatenate([train_abs_error, test_abs_error])

    # 将绝对误差的范围规范化，确保颜色映射从误差最小到最大
    norm = plt.Normalize(vmin=combined_min, vmax=combined_max)
    # norm = plt.Normalize(vmin=combined_abs_error.min(), vmax=combined_abs_error.max())

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制训练集散点图，使用圆点 'o'
    sc_train = ax.scatter(y_train, pred_train, c=train_abs_error, cmap=cmap, norm=norm, alpha=0.8,
                          marker='o', edgecolor='k', label='Training Set')

    # 绘制测试集散点图，使用倒三角形 'v'
    sc_test = ax.scatter(y_test, pred_test, c=test_abs_error, cmap=cmap, norm=norm, alpha=0.8,
                         marker='v', edgecolor='k', label='Test Set')

    # 设置坐标轴范围
    ax.set_xlim(x_axis_start, x_axis_end)
    ax.set_ylim(y_axis_start, y_axis_end)

    # 添加理想拟合的参考线（实线）
    ax.plot([x_axis_start, x_axis_end], [x_axis_start, x_axis_end], 'k-', label='Ideal Fit')

    # 添加平行的误差虚线 (+error_range 和 -error_range 误差范围)
    ax.plot([x_axis_start, x_axis_end], [x_axis_start + error_range, x_axis_end + error_range], 'k--')
    ax.plot([x_axis_start, x_axis_end], [x_axis_start - error_range, x_axis_end - error_range], 'k--')

    # 图形标题和标签
    ax.set_title('Predicted vs True Values with Error Ranges (±{})'.format(error_range))
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')

    # 添加图例，并去除误差线的标注
    ax.legend(loc='upper left', frameon=True)

    # 添加颜色条，表示绝对误差的大小，放在图的左侧并缩小尺寸
    cbar = fig.colorbar(sc_test, ax=ax, shrink=0.7, location='right')
    cbar.set_label('Absolute Error')

    # 显示网格
    ax.grid(False)

    # 调整布局
    plt.tight_layout()
    # 显示图像
    # if file_name != "":
    #     plt.savefig(f'N:/tf_torch\pythonProject/all_cnn_lstm/Figure/{file_name}.png')
    plt.show()


# def plot_loss(train_loss_values, val_loss_values, file_name=""):
#     plt.figure(figsize=(8, 6))
#     red_color = (233 / 255, 143 / 255, 120 / 255)
#     #   plt.plot(train_loss_values, label='Training Loss', linestyle='-', marker='o', markersize=2)
#     #   plt.plot(val_loss_values, label='Validation Loss', linestyle='-', marker='x', markersize=2)
#     plt.plot(train_loss_values, label='Training Loss', linestyle='-', color=red_color, linewidth=1, alpha=1, marker='o',
#              markersize=1)
#     plt.plot(val_loss_values, label='Validation Loss', linestyle='--', linewidth=1, alpha=1, marker='x', markersize=1,
#              markevery=5, dashes=(10, 5))
#
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss Over Epochs")
#     plt.legend()
#     plt.grid(False)  # 添加网格以提高可读性
#     # if file_name != "":
#     plt.savefig('KAN_loss_curve.pdf', bbox_inches='tight', dpi=300)
#     plt.tight_layout()  # 调整布局
#     plt.show()

def plot_loss(train_loss_values, val_loss_values, file_name=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))  # 放大图形尺寸

    blue_color = '#1f77b4'
    orange_color = '#ff7f0e'

    plt.plot(train_loss_values, label='Training Loss', linestyle='-', color=blue_color,
             linewidth=2.5, alpha=0.9, marker='o', markersize=3)
    plt.plot(val_loss_values, label='Validation Loss', linestyle='-', color=orange_color,
             linewidth=2.5, alpha=0.9, marker='x', markersize=3)

    # === 标签与标题 ===
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss Over Epochs", fontsize=17, fontweight='bold')

    # === 刻度字体 ===
    plt.tick_params(axis='both', labelsize=12)

    # === 图例 ===
    plt.legend(fontsize=13, loc='upper right', frameon=True, edgecolor='gray')

    plt.grid(False)
    plt.tight_layout()
    plt.savefig('KAN_loss_curve.pdf', bbox_inches='tight', dpi=300)
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#
def plot_results_ob1(predicted_data, Ytest, r2="", ylabel_name="Airflow speed (m/s)", file_name=""):
    plt.figure(figsize=(10, 8))
    objective1_color = '#1f77b4'  # 深蓝色
    objective2_color = '#ff7f0e'   # 橙色
    plt.plot(predicted_data, label='Predicted values (Testing set)',color='black', linestyle='--', marker='o',markersize=3)
    # plt.plot(Ytest, color=(223 / 255, 143 / 255, 120 / 255), label='True Values (Testing Set)', linestyle='-',
    #         marker='<')
    plt.plot(Ytest, label='True values (Testing set)', linestyle='-',
             marker='x',markersize=6,color=objective1_color)
    plt.xlabel("Sample points")
    plt.ylabel(ylabel_name)
    if r2 != "":
        plt.title(f"The prediction result :\nR²: {r2:.4f}")
    plt.legend(loc='upper left',frameon=True,edgecolor='#1f77b4',framealpha=1)
    plt.ylim([80,230])
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor(objective1_color)
    plt.grid(False)  # 添加网格以提高可读性
    plt.tight_layout()  # 调整布局
    plt.savefig('Prediction 1.pdf', bbox_inches='tight', dpi=300)
    # if file_name != "":
    #     plt.savefig(f'N:/tf_torch\pythonProject/all_cnn_lstm/Figure/{file_name}.png')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_results_ob1_CI(predicted_data, Ytest, r2="", ylabel_name="Airflow speed (m/s)",
                     file_name="", y_std=None, confidence=0.95, scale_factor=1.0):
    """
    绘制预测结果 + 置信区间（可放大）
    参数：
        predicted_data: 模型预测值
        Ytest: 真实值
        r2: R²分数（可选）
        ylabel_name: y轴标题
        file_name: 输出文件名（可选）
        y_std: 每个预测点标准差（不确定性）
        confidence: 置信水平（默认95%）
        scale_factor: 标准差放大倍数（默认1，可设为22000）
    """
    plt.figure(figsize=(10, 8))

    objective_color = '#1f77b4'  # 深蓝色
    plt.plot(predicted_data, label='Predicted values (Testing set)',
             color='black', linestyle='--', marker='o', markersize=3)
    plt.plot(Ytest, label='True values (Testing set)',
             linestyle='-', marker='x', markersize=6, color=objective_color)

    plt.xlabel("Sample points")
    plt.ylabel(ylabel_name)
    if r2 != "":
        # plt.title(f"The prediction result :\nR²: {r2:.4f}")
        plt.title(f"The prediction result :")

    # ==============================
    # 绘制置信区间（95% CI）
    # ==============================
    if y_std is not None:
        # 放大标准差
        y_std_scaled = np.array(y_std) * scale_factor

        # 95% CI = mean ± 1.96 * std
        ci = 1.96 * y_std_scaled
        lower = predicted_data - ci
        upper = predicted_data + ci

        plt.fill_between(
            np.arange(len(predicted_data)),
            lower, upper,
            color=objective_color, alpha=0.2,
            label=f"{int(confidence*100)}% Confidence Interval"
        )

    plt.legend(loc='upper left', frameon=True, edgecolor=objective_color, framealpha=1)
    plt.ylim([140, 250])  # 你原来的纵轴范围
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor(objective_color)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig('Prediction 1 (with CI).pdf', bbox_inches='tight', dpi=300)
    plt.show()



def plot_results_ob2(predicted_data, Ytest, r2="", ylabel_name="Air consumption (L/min)", file_name=""):
    plt.figure(figsize=(10, 8))

    objective1_color = '#1f77b4'  # 深蓝色
    objective2_color = '#ff7f0e'   # 橙色
    plt.plot(predicted_data, label='Predicted values (Testing set)', color='black', linestyle='--', marker='o',markersize=3)
    # plt.plot(Ytest, color=(223 / 255, 143 / 255, 120 / 255), label='True Values (Testing Set)', linestyle='-',
    #         marker='<')
    plt.plot(Ytest, label='True values (Testing set)', linestyle='-',
             marker='x',markersize=6,color=objective2_color)
    plt.xlabel("Sample points")
    plt.ylabel(ylabel_name)
    if r2 != "":
        plt.title(f"The prediction result :\nR²: {r2:.4f}")
    plt.legend(loc='upper left', frameon=True,edgecolor='#ff7f0e',framealpha=1)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor(objective2_color)

        # 设置误差范围变量
    plt.grid(False)  # 添加网格以提高可读性
    plt.tight_layout()  # 调整布局
    plt.savefig('Prediction 2.pdf', bbox_inches='tight', dpi=300)
    # if file_name != "":
    #     plt.savefig(f'N:/tf_torch\pythonProject/all_cnn_lstm/Figure/{file_name}.png')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_results_ob2_CI(predicted_data, Ytest, r2="", ylabel_name="Air consumption (L/min)",
                     file_name="", y_std=None, confidence=0.95, scale_factor=1.0):
    """
    绘制预测结果 + 放大置信区间（支持scale_factor控制）
    参数：
        predicted_data: 模型预测值
        Ytest: 真实值
        r2: R²分数
        ylabel_name: y轴标题
        file_name: 输出文件名
        y_std: 每个预测点标准差（不确定性）
        confidence: 置信水平（默认95%）
        scale_factor: 标准差放大倍数，默认1，可设为22000增强可视化
    """
    plt.figure(figsize=(10, 8))

    # --- 基本样式 ---
    objective_color = '#ff7f0e'  # 橙色
    plt.plot(predicted_data, label='Predicted values (Testing set)',
             color='black', linestyle='--', marker='o', markersize=3)
    plt.plot(Ytest, label='True values (Testing set)',
             linestyle='-', marker='x', markersize=6, color=objective_color)

    plt.xlabel("Sample points")
    plt.ylabel(ylabel_name)
    if r2 != "":
        # plt.title(f"The prediction result :\nR²: {r2:.4f}")
        plt.title(f"The prediction result :")

    # --- 绘制置信区间 ---
    if y_std is not None:
        # 放大标准差
        y_std_scaled = np.array(y_std) * scale_factor

        # 计算置信区间（mean ± 1.96*std）
        ci = 1.96 * y_std_scaled
        lower = predicted_data - ci
        upper = predicted_data + ci

        plt.fill_between(np.arange(len(predicted_data)),
                         lower, upper,
                         color=objective_color, alpha=0.2,
                         label=f"{int(confidence*100)}% Confidence Interval")

    # --- 样式与美化 ---
    plt.legend(loc='upper left', frameon=True, edgecolor=objective_color, framealpha=1)
    plt.ylim([5, 35])
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor(objective_color)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig('Prediction 2 (with CI).pdf', bbox_inches='tight', dpi=300)
    plt.show()




# import skill_metrics as sm


# def plot_taylor_diagram_sm(sd, rmsd, cc, labels):
#     fig = plt.figure(figsize=(10, 8))
#
#     markercolor = ['o', 's', '^', 'D']
#     markercolor = ['b', 'g', 'r', 'c', 'm', 'y']
#     # 绘制泰勒图
#     # sm.taylor_diagram(sd, rmsd, cc,
#     #                   locationColorBar='EastOutside',
#     #                   colRMS='m', styleRMS=':', widthRMS=1.5,
#     #                   colSTD='k', styleSTD='-', widthSTD=1.0,
#     #                   colCOR='k', styleCOR='--', widthCOR=1.7)
#     print(sd, rmsd, cc)
#     print(labels)
#
#     sm.taylor_diagram(sd, rmsd, cc, markerLabel=labels,
#                       markerLabelColor='k',
#                       markerLegend='on',
#                       styleOBS='-', colOBS='r', markerobs='o', markerSize=6,
#                       tickRMSangle=115, showlabelsRMS='on',
#                       titleRMS='on', titleOBS='Ref')
#
#     # 设置标题和图例
#     # plt.title("Taylor Diagram of Model Performance", fontsize=10)
#     plt.xlabel('Observation', x=0.4, fontweight='bold', font='Times New Roman', fontsize=12)
#     plt.ylabel('Standard Deviation', x=-1, fontweight='bold', font='Times New Roman', fontsize=12)
#     # 获取图例对象并去除边框
#     legend = plt.gca().get_legend()
#     if legend is not None:
#         legend.set_frame_on(False)  # 去除图例的边框
#     # plt.savefig('N:/tf_torch\pythonProject/all_cnn_lstm/Figure/Figure_plotTalyor_grm.png')
#     plt.grid(False)  # 添加网格以提高可读性
#     plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_boxplot_from_numpy(data, n_groups=3, title="Boxplot", xlabel=None, ylabel=None,
                            palette="Set2", showfliers=True, width=0.3, figsize=(8, 6),
                            save_path=None):
    """
    绘制箱线图的通用函数，支持传入 NumPy 数组。根据数据自动划分为指定数量的组。

    参数:
    - data: np.ndarray 数值型数据，形状为 (n_samples, ) 或 (n_samples, n_features)。
    - n_groups: int, 要将数据分成的组数，默认是 2。
    - title: str 图的标题。
    - xlabel: str, optional 横轴标签，默认为 "Category"。
    - ylabel: str, optional 纵轴标签，默认为 "Values"。
    - palette: str or list 配色方案，默认使用 "Set2"。
    - showfliers: bool 是否显示离群点，默认显示。
    - width: float 箱线图的宽度，默认值为 0.8。
    - figsize: tuple 图的尺寸，默认值为 (8, 6)。
    - save_path: str, optional 如果提供路径，将图像保存到该路径。

    返回:
    - None
    """
    # 确保 data 是一维数组
    if len(data.shape) == 2:
        data = data.flatten()  # 如果是二维数组，展平

    n_samples = len(data)

    # 计算每个组的样本数（按比例分配）
    group_sizes = [int(n_samples * (1.0 / n_groups))] * n_groups

    # 对于剩余样本，均匀分配到各组
    remainder = n_samples - sum(group_sizes)
    for i in range(remainder):
        group_sizes[i] += 1

    # 创建每个组的标签
    categories = np.concatenate([np.repeat(f"Group {i + 1}", size) for i, size in enumerate(group_sizes)])

    # 将数据组织为 DataFrame
    df = pd.DataFrame({"Values": data, "Category": categories})

    plt.figure(figsize=figsize)

    # 绘制箱线图，按分类变量进行分组
    sns.boxplot(x="Category", y="Values", data=df, palette=palette, showfliers=showfliers, width=width)

    # 设置标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel if xlabel else "Category", fontsize=14)
    plt.ylabel(ylabel if ylabel else "Values", fontsize=14)

    # 调整布局并保存图像
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# 定义Taylor图类
class TaylorDiagram:
    def __init__(self, refstd, fig=None, rect=111, label='Reference', srange=(0, 2),
                 linestyle1=None, linestyle2=None, extend=False, **kwargs):
        self.refstd = refstd  # 参考模型的标准差
        self.fig = fig or plt.figure()  # 创建fig
        self.ax = self.fig.add_subplot(rect, polar=True, label=label, **kwargs)  # 子图
        self.srange = srange  # 图形边界
        self.rect = rect
        if extend:  # 定义是否有r为负值的情况
            self.tmax = np.pi
        else:
            self.tmax = np.pi / 2

        # 限制角度范围为四分之一圆 (0 到 90 度, 也就是 π/2 弧度)
        self.ax.set_ylim(0, srange[1] * self.refstd)  # 0-2
        self.ax.set_xlim(0, self.tmax)  # 限制到四分之一圆

        # 手动输入需要确定相关系数(若有负相关需要再次设置)
        corr_ticks = [1.47, 1.369, 1.266, 1.159, 1.047, 0.927, 0.795, 0.6435, 0.451, 0.3176, 0.1415, 0]
        corr_labels = np.round(np.cos(corr_ticks), 2)  # cos(theta) 对应相关系数
        self.ax.set_thetagrids(np.degrees(corr_ticks), labels=corr_labels)
        # 单独设置x网格线样式
        self.ax.xaxis.grid(color='k', linestyle=linestyle1, linewidth=1)
        # 单独设置y网格线样式
        self.ax.yaxis.grid(color='k', linestyle=linestyle2, linewidth=1)

        # 设置标准差刻度
        self.ax.set_rgrids(np.arange(srange[0] * self.refstd, (srange[1] + 0.1) * self.refstd,
                                     0.5 * self.refstd), angle=0)
        list = np.round(np.arange(srange[0] + 0.5, srange[1] + 0.1, 0.5) * self.refstd, 2)
        for i in range(4):
            self.ax.text(np.pi / 2, np.round(0.5 * (i + 1) * self.refstd, 2), f'{list[i]}\n', ha='center', va='center',
                         fontsize=14, rotation=90)
        # 自定义CRMSE标值
        a = [0.4, 0.79, 1.1, 1.43]
        b = [1.25, 1.4, 1.65, 1.88]
        rot = [0, 5, 15, 25]
        for i in range(4):
            self.ax.text(a[i], np.round(b[i] * self.refstd, 2), f'{list[i]}\n', ha='center', va='center',
                         fontsize=14, rotation=rot[i], c='red', fontweight='bold')

        # 添加"Correlation Coefficient"标签
        text_n = 'Correlation Coefficient'  # 标签文本
        n_text = len(text_n)
        theta = np.linspace(np.pi / 8 * 3, np.pi / 8, n_text)  # 改为60度范围
        r = np.full_like(theta, self.refstd + 1.2 * self.refstd)  # 稍微向外移动标签
        for i in range(n_text):
            self.ax.text(theta[i], r[i], text_n[i], ha='center', va='center',
                         rotation=np.degrees(-theta[n_text - 1 - i]), fontsize=16, fontweight='bold')
        # 添加"CRMSE"标签
        text_n = 'CRMSE'  # 标签文本
        n_text = len(text_n)
        thetaa = [0.4476, 0.5325, 0.565, 0.5735, 0.56]
        r = [0.58, 0.68, 0.785, 0.895, 0.99]
        theta_n = [1.1, 0.86, 0.5925, 0.532, 0.4005]

        for i in range(n_text):
            self.ax.text(thetaa[i], r[i] * self.refstd, text_n[i], ha='center', va='center',
                         rotation=np.degrees(theta_n[i]), fontsize=16, fontweight='bold')

        # 绘制1.0obs的标准弧线并加粗
        r_1 = np.full(100, 1.0) * self.refstd  # 半径为1.0
        theta = np.linspace(0, np.pi / 2, 100)
        self.ax.plot(theta, r_1, 'b-', linewidth=1.5)  # 设置弧线为红色，粗2个单位
        self.ax.text(-0.2, 0.8 * self.refstd, 'Observation', c='blue', fontweight='bold', font='Times New Roman',
                     fontsize=14)  # 设置弧线为红色，粗2个单位

        self.ax.set_ylabel('Standard Deviation', fontweight='bold', font='Times New Roman', fontsize=14)
        self.ax.yaxis.set_label_coords(-0.05, 0.5)  # 向左移动标签

    def add_sample(self, stddev, corrcoef, marker=None, color=None, label=None, markersize=None):  # 打点
        r = stddev
        theta_plt = np.arccos(np.clip(corrcoef, -1, 1))  # 确保相关系数在[-1, 1]范围内
        self.ax.plot(theta_plt, r, marker=marker, color=color, label=label, markersize=markersize)  # 打点模式
        # 打点观测数据
        self.ax.plot(np.arccos(1), self.refstd, marker='^', color='r', label='Obs', markersize=16, zorder=2)

    def add_contours(self, **kwargs):  # 画RMSE
        rs, ts = np.meshgrid(np.linspace(self.srange[0] * self.refstd,
                                         self.srange[1] * self.refstd, 100), np.linspace(0, self.tmax, 100))
        # rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))
        rms = np.sqrt(self.refstd ** 2 + rs ** 2 - 2 * self.refstd * rs * np.cos(ts))
        contour = self.ax.contour(ts, rs, rms, levels=np.round([0.5 * float(self.refstd), 1.0 * float(self.refstd),
                                                                1.5 * float(self.refstd), 2.0 * float(self.refstd)], 2),
                                  **kwargs)

    def __str__(self):  # 提取模型参数
        return f" ---------------Parameters--------------\n" \
               f" Range：{self.tmax} STD(obs)：{self.refstd} \n " \
               f"Boundary：{self.srange} Ax_Position :{self.rect}"


# ----绘制Taylor图的函数，支持传入samples和观测std---
def plot_taylor_diagram(samples, obs_std):
    fig = plt.figure(figsize=(8, 6))  # 创建绘图
    # 输入标准化后的std，即观测std=1
    dia = TaylorDiagram(obs_std, fig=fig, linestyle1='-.', linestyle2='--')  # refstd,fig

    # 添加样本数据
    for stddev, corrcoef, marker, color, label, size in samples:
        dia.add_sample(stddev, corrcoef, marker=marker, color=color, label=label, markersize=size)

    # 添加等值线（设置RMSE线的ls）
    dia.add_contours(colors='red', linestyles='--')  # RMSE

    # 添加图例，显示marker和颜色
    handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=10, label=label)
               for _, _, marker, color, label, _ in samples]
    # 在handles的第一个位置插入Obs图例
    handles.insert(0, plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='r', markersize=10, label='Obs'))
    # 调整图例位置 (1.42, 1.1)x和y位置
    plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.42, 1.1), frameon=False, fontsize=12)
    # y轴标签
    plt.tight_layout()
    plt.show()
