import numpy as np
import pandas as pd
import random
def preconditioning(df):
    
    df.loc[-1]=df.columns

    df.columns=[f'{i+2014}' for i in range(10)]
    df.sort_index(inplace=True) 
    new_cow =[ '2014' , '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

    new_index = ['Supply chain disruption', 'Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']

    # 将新的索引设置为 DataFrame 的索引
    df.index = new_index
    
    df=df*100
    
    return df

# 假设给定的父节点数据和子节点数据
np.random.seed(42)
n_samples = 200
n_parent_nodes = 7

# 随机生成父节点数据 (每个父节点0或1)
rand=np.random.randint(2014, 2024)
df = pd.read_excel('result.xlsx')

df=preconditioning(df)
list1=[]
print(df)
for i in range(n_samples):
    data_temp=df.copy()
    
    ran=random.random()
    data_temp[data_temp>ran]=int(1)
    data_temp[data_temp<ran]=int(0)
    data1=data_temp[f'{rand}']
    list1.append(data1.to_list()[1:])
    
m=np.array(list1)
X_ = m.astype(int)
print(X_)
X = np.random.randint(0, 2, size=(n_samples, n_parent_nodes))
print(X)
# 假设子节点的生成与父节点的关系有一定的规则，这里是一个简化的例子
# 子节点 Y = f(X) + 噪声（这里只是一个简单的例子，实际上你需要根据实际问题来建立子节点和父节点之间的关系）
prob_y_given_x = 0.7  # 假设当父节点是全1时，子节点为1的概率是0.7

# 根据父节点的不同，生成子节点数据
y = np.array([1 if np.random.rand() < prob_y_given_x else 0 for _ in range(n_samples)])

# 初始化条件概率
# 假设初始条件概率是均匀分布
theta_init = np.random.rand(2**n_parent_nodes)  # 7个父节点，条件概率有2^7种组合

# E步：计算隐含变量的期望值
def e_step(X, theta):
    # 计算每个父节点组合的条件概率
    prob = np.zeros((X.shape[0], 2**n_parent_nodes))
    for i in range(X.shape[0]):
        idx = int("".join(map(str, X[i])), 2)  # 将父节点组合转换为整数索引
        prob[i, idx] = theta[idx]
    
    # 归一化概率
    prob /= prob.sum(axis=1, keepdims=True)
    
    return prob

# M步：更新参数
def m_step(X, y, prob):
    new_theta = np.zeros(2**n_parent_nodes)
    for i in range(2**n_parent_nodes):
        # 更新参数为父节点组合为i的条件下子节点为1的期望值
        indices = np.where(np.array([int("".join(map(str, X[j])), 2) == i for j in range(X.shape[0])]))[0]
        new_theta[i] = np.mean(y[indices])
    
    return new_theta

# EM算法迭代
def em_algorithm(X, y, theta_init, n_iter=100):
    theta = theta_init
    for _ in range(n_iter):
        prob = e_step(X, theta)
        theta = m_step(X, y, prob)
    
    return theta

# 运行EM算法
theta_est = em_algorithm(X, y, theta_init, n_iter=100)

# 打印最终估计的条件概率
print("Estimated conditional probabilities for the child node given parent nodes:")
for i in range(2**n_parent_nodes):
    print(f"Parent combination {bin(i)[2:].zfill(n_parent_nodes)}: {theta_est[i]:.4f}")
