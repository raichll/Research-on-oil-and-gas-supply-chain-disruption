import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random

# E步：计算隐含变量的期望值
def e_step(X, theta):
    prob = np.zeros((X.shape[0], 2**n_parent_nodes))
    
    # 计算每个父节点组合的条件概率
    for i in range(X.shape[0]):
        idx = int("".join(map(str, X[i])), 2)  # 将父节点组合转换为整数索引
        prob[i, idx] = theta[idx]
    
    # 归一化概率，确保每一行的总和为1
    prob_sum = prob.sum(axis=1, keepdims=True)
    
    # 防止除以零，避免 NaN
    prob_sum[prob_sum == 0] = 1  # 将和为0的行设置为1，避免除以0
    
    prob /= prob_sum  # 归一化
    return prob

# M步：更新参数
def m_step(X, y, prob):
    new_theta = np.zeros(2**n_parent_nodes)
    
    for i in range(2**n_parent_nodes):
        # 获取父节点组合为i的样本
        indices = np.where(np.array([int("".join(map(str, X[j])), 2) == i for j in range(X.shape[0])]))[0]
        
        # 对每个父节点组合，计算子节点为1的期望值
        new_theta[i] = np.mean(y[indices]) if len(indices) > 0 else 0  # 处理空索引
        
    return new_theta

# EM算法迭代
def em_algorithm(X, y, theta_init, n_iter=100):
    theta = theta_init
    for iteration in range(n_iter):
        prob = e_step(X, theta)
        theta = m_step(X, y, prob)
        
        # 打印调试信息，查看参数变化
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: theta = {theta[:10]}...")  # 打印前10个theta值
    
    return theta

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


# 模拟数据
np.random.seed(42)
n_samples = 200
n_parent_nodes = 7


# 随机生成父节点数据 (每个父节点0或1)

df = pd.read_excel('result.xlsx')

df=preconditioning(df)
list1=[]
print(df)
# 初始化 MinMaxScaler
#scaler = MinMaxScaler()

# 对每列进行归一化
#df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
for i in range(n_samples):
    data_temp=df.copy()
    rand=np.random.randint(2014, 2024)
    data1=data_temp[f'{rand}']
    
    df_normalized_manual = (data1 - data1.min()) / (data1.max() - data1.min())
    ran=random.random()
    print(df_normalized_manual)
    df_normalized_manual[df_normalized_manual>ran]=int(1)
    df_normalized_manual[df_normalized_manual<ran]=int(0)
    print()
    list1.append(df_normalized_manual.to_list()[1:])
    
m=np.array(list1)
X = m.astype(int)
print(X)
#print(X_)
#X = np.random.randint(0, 2, size=(n_samples, n_parent_nodes))

# 随机生成父节点数据 (每个父节点0或1)
#X = np.random.randint(0, 2, size=(n_samples, n_parent_nodes))

# 假设子节点的生成与父节点的关系有一定的规则，这里是一个简化的例子
prob_y_given_x = 0.7  # 假设当父节点是全1时，子节点为1的概率是0.7

# 根据父节点的不同，生成子节点数据
y = np.array([1 if np.random.rand() < prob_y_given_x else 0 for _ in range(n_samples)])
print(y)
# 初始化条件概率
# 7个父节点，条件概率有2^7种组合，初始化为均匀分布
theta_init = np.random.rand(2**n_parent_nodes)
theta_init /= np.sum(theta_init)  # 归一化到[0, 1]范围
# 运行EM算法
theta_est = em_algorithm(X, y, theta_init, n_iter=20)

# 打印最终估计的条件概率
print("\nEstimated conditional probabilities for the child node given parent nodes:")
for i in range(2**n_parent_nodes):
    print(f"Parent combination {bin(i)[2:].zfill(n_parent_nodes)}: {theta_est[i]:.4f}")
