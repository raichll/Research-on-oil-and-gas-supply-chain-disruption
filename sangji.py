import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 假设联合概率矩阵 P(A, B)
P_AB = np.array([[0.1, 0.2, 0.3],
                 [0.4, 0.5, 0.6]])

# 计算边际概率 P(B)
P_B = P_AB.sum(axis=0)

# 计算条件概率 P(A|B)
P_A_given_B = P_AB / P_B

# 创建条件概率表格
df = pd.DataFrame(P_A_given_B, index=['a1', 'a2'], columns=['b1', 'b2', 'b3'])

# 绘制条件概率表
plt.figure(figsize=(6, 4))
plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center')
plt.axis('off')
plt.title("Conditional Probability Table P(A|B)")
plt.show()