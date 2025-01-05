from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 定义贝叶斯网络结构
nodes = ['Supply chain disruption', 'Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']
edges = [
    ('Technical', 'Supply chain disruption'),
    ('Economic', 'Supply chain disruption'),
    ('Social', 'Supply chain disruption'),
    ('Political', 'Supply chain disruption'),
    ('Safety', 'Supply chain disruption'),
    ('Environmental', 'Supply chain disruption'),
    ('Legal', 'Supply chain disruption')
]

# 创建贝叶斯网络
model = BayesianNetwork(edges)

# 手动构建 NetworkX 图
nx_graph = nx.DiGraph(model.edges())

# 设置节点布局
# 子节点放中间，其余节点围绕
pos = nx.shell_layout(nx_graph, nlist=[['Supply chain disruption'], 
                                       ['Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']])

# 绘制贝叶斯网络
plt.figure(figsize=(10, 8))

# 为 'Supply chain disruption' 设置较大的节点大小
#node_sizes = [1000 if node == 'Supply chain disruption' else 500 for node in nx_graph.nodes()]

# 调整字体大小
font_size = 14  # 增大字体大小

# 绘制节点时使用椭圆形
for node, (x, y) in pos.items():
    # 椭圆的大小（宽度和高度）
    width = 1.0 if node == 'Supply chain disruption' else 0.8
    height = 0.5 if node == 'Supply chain disruption' else 0.4
    # 绘制椭圆形节点
    ellipse = mpatches.Ellipse((x, y), width, height, color='lightblue', ec='black', lw=2, clip_on=True)
    plt.gca().add_patch(ellipse)
    
    # 将标签放在椭圆内
    plt.text(x, y, node, fontsize=font_size + 2 if node == 'Supply chain disruption' else font_size,
             ha='center', va='center', fontweight='bold', color='black')

# 绘制边
nx.draw_networkx_edges(nx_graph, pos, edge_color='gray', arrowsize=15)

# 绘制标题
plt.title("Bayesian Network: Supply Chain Disruption", fontsize=18)
plt.axis('off')  # 不显示坐标轴
plt.show()
