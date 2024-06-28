import matplotlib.pyplot as plt
import pandas as pd

# 读取数据文件
data1 = pd.read_csv("SELD\\RESULTS\\DualQSELD-TCN-PHI-S1_BN_RF287_10RB_8ch_training_metrics_original.csv", header=None, names=["Epoch", "Train Loss", "Validation Loss"])
data2 = pd.read_csv("SELD\\RESULTS\\DualQSELD-TCN-PHI-S1_BN_RF287_10RB_8ch_training_metrics_update.csv", header=None, names=["Epoch", "Train Loss", "Validation Loss"])

""" # 绘制第一个模型的训练和验证损失曲线
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(data1["Epoch"], data1["Train Loss"], label="Train Loss")
plt.plot(data1["Epoch"], data1["Validation Loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 1: Training and Validation Loss")
plt.legend()

# 绘制第二个模型的训练和验证损失曲线
plt.subplot(1, 2, 2)
plt.plot(data2["Epoch"], data2["Train Loss"], label="Train Loss")
plt.plot(data2["Epoch"], data2["Validation Loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 2: Training and Validation Loss")
plt.legend()

plt.tight_layout()
plt.show() """

plt.figure(figsize=(10, 6))

plt.plot(data1["Epoch"], data1["Train Loss"], label="Model 1 Train Loss--original", color='b', linestyle='-')
plt.plot(data1["Epoch"], data1["Validation Loss"], label="Model 1 Validation Loss--original", color='b', linestyle='--')
plt.plot(data2["Epoch"], data2["Train Loss"], label="Model 2 Train Loss--update", color='r', linestyle='-')
plt.plot(data2["Epoch"], data2["Validation Loss"], label="Model 2 Validation Loss--update", color='r', linestyle='--')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss for Two Models")
plt.legend()

plt.show()


""" 
模型1的训练损失迅速下降并趋于稳定，但验证损失波动较大，显示模型在训练数据上表现良好，但在验证数据上的泛化能力欠佳。
模型2在训练和验证损失上的下降趋势较为平滑和稳定，验证损失波动较小，表明模型在训练数据和验证数据上都有较好的表现，泛化能力更强。 
"""