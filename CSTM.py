        
import matplotlib.pyplot as plt      
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
    df[df>1]=1
    return df

def gm11(data, predict_length=10):
    """
    GM(1,1) 灰色预测模型
    :param data: 输入数据（时间序列，list 或 numpy array）
    :param predict_length: 预测长度
    :return: 原始数据、拟合值、预测值
    """
    data = np.array(data)
    n = len(data)

    # 1-AGO 累加生成序列
    agg_data = np.cumsum(data)
    
    # 构造 B 和 Y 矩阵
    B = np.zeros((n - 1, 2))
    for i in range(n - 1):
        B[i, 0] = -0.5 * (agg_data[i] + agg_data[i + 1])
        B[i, 1] = 1
    Y = data[1:]

    # 求解参数 a 和 b
    coef = np.linalg.inv(B.T @ B) @ B.T @ Y
    a, b = coef[0], coef[1]
 
    # 构造预测公式
    def predict(k):
        return (data[0] - b / a) * np.exp(-a * (k - 1)) + b / a
    def fittd(k):
        if k==1:
            return data[0]
        else:
            return (data[0] - b / a) *(1-np.exp(a)) *np.exp(-a * (k - 1)) 
    # 计算拟合值
    fitted = np.array([fittd(i) for i in range(1, n + 1)])
    
    residual = data - fitted

    # 预测未来值
    predicted = np.array([predict(i) for i in range(n + 1, n + 1 + predict_length)])

    return data, fitted, predicted, residual

def roulette_wheel_selection(probabilities):
    # 计算累积概率
    cumulative_probabilities = np.cumsum(probabilities)
    
    # 生成一个[0,1)之间的随机数
    rand_val = np.random.random()
    
    # 根据随机数选择0或1
    for i, cumulative_prob in enumerate(cumulative_probabilities):
        if rand_val < cumulative_prob:
            return 1  # 返回1表示转化为1
    return 0  # 如果没有匹配，返回0

def cstm(states):
    
    num_states = 2  # 状态空间只有0和1
    transition_counts = np.zeros((num_states, num_states))  # 用来记录转移次数

    # 计算转移次数
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transition_counts[current_state][next_state] += 1

    # 计算转移概率矩阵
    row_sums = transition_counts.sum(axis=1)

# 防止除以零，使用 np.where 来避免行和为零的情况
    transition_matrix = np.divide(transition_counts.T, row_sums, where=row_sums != 0).T
    return transition_matrix

if __name__ == "__main__":
    df = pd.read_excel('result.xlsx')

    df=preconditioning(df)
    
    print(df)
    new_index = ['Supply chain disruption', 'Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']
    li=[]
    f=[]
    for i in range(8):
        # 测试数据
        data = df.loc[new_index[i]].to_list()
        
        #print(data)    # 使用 GM(1,1) 模型预测未来 10 个值
        original, fitted, predicted, residual = gm11(data, predict_length=10)
        #print(fitted)
        f.append(fitted)
        res_all = np.zeros((2, 2))
        res_all2= np.zeros((2, 2))
        t=1000
        for j in range(t):
            list_original=[]
            for i in range(10):
                if random.random() <= original[i]:
                    list_original.append(1)
                else :
                    list_original.append(0)
                
            res=cstm(list_original)
            res_all+=(res)/t
                    
            list_fitted=[]
            for i in range(10):
                if random.random() <= fitted[i]:
                    list_fitted.append(1)
                else :
                    list_fitted.append(0)
        
            res2=cstm(list_fitted)
            res_all2+=(res2)/t
        
        li.append(res_all2)
    
    print(li)
    df1=df.T.copy()
    
    df2=pd.DataFrame(f,index=new_index).T.copy()
    df2.index=df1.index
    # 可视化
    #years = list(range(2014, 2024))  # 2014到2023年
    # 绘制第一个 DataFrame
    colors = {
    'Supply chain disruption': '#1f77b4',  # 蓝色
    'Technical': '#ff7f0e',  # 橙色
    'Economic': '#2ca02c',  # 绿色
    'Social': '#d62728',  # 红色
    'Political': '#9467bd',  # 紫色
    'Safety': '#8c564b',  # 棕色
    'Environmental': '#e377c2',  # 粉色
    'Legal': '#7f7f7f',  # 灰色
}
    
# 绘制第一个 DataFrame 的折线图
    for column in df1.columns:
        
        plt.plot(df1.index, df1[column], label=f'df1: {column}', color=colors[column], linestyle='-', marker='o')

    # 绘制第二个 DataFrame 的折线图
    for column in df2.columns:
        plt.plot(df2.index, df2[column], label=f'df2: {column}', color=colors[column], linestyle='--', marker='x')

    #plt.plot(df, label='original', color='blue')
    #plt.plot(li,  label='fitted', color='green')
    plt.legend()
    plt.xlabel("year")
    plt.ylabel("value")
    #plt.title("gery fitted")
    plt.grid()
    plt.show()
