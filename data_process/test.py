# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# print("000")
# # Load the CSV file
# file_path = r"g:/Results/EmoGen_Reproduction/data_process/evaluation_emoset.csv"
# columns_to_load = ["Filename", "Emotion_score"]

# try:
#     df = pd.read_csv(file_path, usecols=columns_to_load)
#     print("010")
# except Exception as e:
#     print(f"Error loading CSV: {e}")
#     exit()

# # 删除缺失值
# if df["Emotion_score"].isnull().sum() > 0:
#     print(
#         f"Found {df['Emotion_score'].isnull().sum()} missing values in Emotion_score. Dropping..."
#     )
#     df = df.dropna(subset=["Emotion_score"])

# # 分箱操作
# bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

# try:
#     df["Confidence_Interval"] = pd.cut(
#         df["Emotion_score"], bins=bins, labels=labels, include_lowest=True
#     )
#     print("011")
# except Exception as e:
#     print(f"Error during binning: {e}")
#     exit()

# # 统计每个区间的样本数量
# interval_counts = df["Confidence_Interval"].value_counts(sort=False)

# # 检查统计结果
# if interval_counts.empty:
#     print("No data available in interval_counts. Exiting...")
#     exit()

# print("Interval counts:")
# print(interval_counts)

# # 绘制柱状图
# try:
#     plt.figure(figsize=(10, 6))
#     plt.bar(
#         [str(i) for i in interval_counts.index], interval_counts.values, color="skyblue"
#     )
#     print("101")
# except Exception as e:
#     print(f"Error during bar plot creation: {e}")
#     exit()

# # 自定义图表
# plt.title("Distribution of Emotion Scores", fontsize=14)
# plt.xlabel("Confidence Interval (Emotion Score)", fontsize=12)
# plt.ylabel("Number of Samples", fontsize=12)
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()

# # 保存图表
# output_path = r"G:\Results\EmoGen_Reproduction\data_process\emotion_scores_barplot.png"
# try:
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保路径存在
#     plt.savefig(output_path, dpi=300)
#     print(f"Plot saved to {output_path}")
# except Exception as e:
#     print(f"Error saving plot: {e}")
#     exit()


import matplotlib.pyplot as plt

# 数据
confidence_intervals = [
    "0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0",
]
counts = [0, 1, 262, 2294, 6908, 10669, 10401, 11250, 15292, 61025]

# 创建柱状图
plt.figure(figsize=(10, 6))
plt.bar(confidence_intervals, counts, color="skyblue")

# 自定义图表
plt.title("Distribution of Emotion Scores", fontsize=16)
plt.xlabel("Confidence Interval (Emotion Score)", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.xticks(rotation=45, ha="right")  # 旋转 x 轴标签
plt.tight_layout()

# 保存并显示图表
output_path = "emotion_scores_barplot.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
plt.show()
