import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
data=np.array([
    [1.        , 0.90869258, 0.83295899, 0.83445741, 0.8430973 , 0.76875165, 0.68169065, 0.71661943, 0.70950409, 0.71471331],
    [0.89651882, 0.40609774, 0.35718418, 0.35283761, 0.35252134, 0.3497862 , 0.34692347, 0.3480203 , 0.34784113, 0.34804387],
    [0.44263248, 0.46647911, 0.4737269 , 0.47273128, 0.47047402, 0.47916254, 0.49006795, 0.48330905, 0.48504716, 0.48622826],
    [0.52027004, 0.34806956, 0.33825198, 0.33942161, 0.34305491, 0.32607874, 0.3051999 , 0.31529083, 0.31295451, 0.31295096],
    [0.60690609, 0.52923321, 0.54815174, 0.5466194 , 0.54457929, 0.55455711, 0.5665875 , 0.56009741, 0.56162654, 0.56215506],
    [0.49517968, 0.26691323, 0.26138989, 0.26232854, 0.2666955 , 0.24797697, 0.22462933, 0.23642077, 0.23360999, 0.23321737],
    [0.45781711, 0.21623213, 0.22084592, 0.2206736 , 0.22514952, 0.20834639, 0.18691862, 0.19835004, 0.19551602, 0.19468131],
    [0.48561437, 0.52719697, 0.51861753, 0.51147798, 0.49887502, 0.4653427 , 0.43241273, 0.42948573, 0.4314672 , 0.44439294]
])
new_index = ['Supply chain disruption', 'Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']
                                 
years=[i+2014 for i in range(10)]

# 创建颜色映射
colormap = cm.get_cmap("viridis", 8)  # 使用 'viridis' 颜色库生成 8 种颜色

# 创建画布和子图
fig, axes = plt.subplots(8, 1, figsize=(12, 12), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.5, hspace=0.15)

# 绘制每一行数据，分配不同颜色
for i, ax in enumerate(axes.flatten()):
    ax.plot(years, data[i], marker='o', color=colormap(i), label=f" {new_index[i]}")
    #ax.set_title(f" {new_index[i]}", fontsize=10)
    ax.fill_between(years, data[i], 1, color='yellow', alpha=0.3)  # 填充上方区域
    ax.fill_between(years, data[i], 0, color='blue', alpha=0.3)    # 填充下方区域
    ax.set_xlim(2014, 2023)  # 限制横坐标范围
    ax.set_xticks(years)  # 明确设置刻度
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    
# 设置公共标签
fig.text(0.5, 0.04, "Year", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Probability", va="center", rotation="vertical", fontsize=12)

# 调整布局
#plt.suptitle("Probability Trends for Each Data Series (2014-2023)", fontsize=14, y=0.95)
plt.show()
