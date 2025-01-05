        
        
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

def ccpt(data):
    

    # 初始化参数
    P_A_0 = 0.5  # P(A=0)
    P_A_1 = 0.5  # P(A=1)
    P_B_given_A_0 = 0.6  # P(B=1 | A=0)
    P_B_given_A_1 = 0.7  # P(B=1 | A=1)
    P_A_0_given_B_1 = 0  # 默认值，防止未赋值的错误
    P_A_1_given_B_1 = 0  # 默认值，防止未赋值的错误

    # EM算法的迭代次数
    num_iterations = 10

    # 执行EM算法
    for iteration in range(num_iterations):
        #print(f"Iteration {iteration + 1}:")
        
        # E步：计算期望值（填补缺失数据）
        missing_A_0 = 0  # 缺失A时，B=0的期望A=0概率
        missing_A_1 = 0  # 缺失A时，B=1的期望A=1概率
        
        # 计算当前缺失值的后验概率
        for i in range(len(data)):
            if np.isnan(data[i, 0]):  # A缺失的行
                B_val = data[i, 1]
                if B_val == 0:
                    P_A_0_given_B_0 = (P_A_0 * (1 - P_B_given_A_0)) / ((P_A_0 * (1 - P_B_given_A_0)) + (P_A_1 * (1 - P_B_given_A_1)))
                    P_A_1_given_B_0 = 1 - P_A_0_given_B_0
                    missing_A_0 += P_A_0_given_B_0
                    missing_A_1 += P_A_1_given_B_0
                elif B_val == 1:
                    P_A_0_given_B_1 = (P_A_0 * P_B_given_A_0) / ((P_A_0 * P_B_given_A_0) + (P_A_1 * P_B_given_A_1))
                    P_A_1_given_B_1 = 1 - P_A_0_given_B_1
                    missing_A_0 += P_A_0_given_B_1
                    missing_A_1 += P_A_1_given_B_1
            else:
                # A有值的情况，直接统计
                if data[i, 0] == 0:
                    missing_A_0 += 1
                elif data[i, 0] == 1:
                    missing_A_1 += 1

        # M步：更新参数
        # 更新 P(A)
        P_A_0 = missing_A_0 / len(data)
        P_A_1 = missing_A_1 / len(data)
        
        # 更新 P(B|A)
        # 统计 P(B=1 | A=0) 和 P(B=1 | A=1)
        count_B_1_given_A_0 = np.sum((data[:, 0] == 0) & (data[:, 1] == 1)) + missing_A_0 * P_A_0_given_B_1
        count_A_0 = np.sum(data[:, 0] == 0) + missing_A_0
        P_B_given_A_0 = count_B_1_given_A_0 / count_A_0 if count_A_0 > 0 else 0

        count_B_1_given_A_1 = np.sum((data[:, 0] == 1) & (data[:, 1] == 1)) + missing_A_1 * P_A_1_given_B_1
        count_A_1 = np.sum(data[:, 0] == 1) + missing_A_1
        P_B_given_A_1 = count_B_1_given_A_1 / count_A_1 if count_A_1 > 0 else 0
        
        # 打印每次迭代的参数
        '''print(f"P(A=0) = {P_A_0:.4f}, P(A=1) = {P_A_1:.4f}")
        print(f"P(B=1 | A=0) = {P_B_given_A_0:.4f}, P(B=1 | A=1) = {P_B_given_A_1:.4f}")
        print("-" * 40)'''
        #return [P_B_given_A_0,1-P_B_given_A_0,P_B_given_A_1,1-P_B_given_A_1]
    # 最终结果
    '''print("Final Parameters:")
    print(f"P(A=0) = {P_A_0:.4f}, P(A=1) = {P_A_1:.4f}")
    print(f"P(B=1 | A=0) = {P_B_given_A_0:.4f}, P(B=1 | A=1) = {P_B_given_A_1:.4f}")'''
    
    return [P_B_given_A_0,1-P_B_given_A_0,P_B_given_A_1,1-P_B_given_A_1]


if __name__ == "__main__":
    df = pd.read_excel('result.xlsx')

    df=preconditioning(df)
    
  
    
    #print(df)
    t=100
    
    new_index = ['Supply chain disruption', 'Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']
    res_all = np.zeros((7, 4))
    
    for i in range(t):
        data1=df.copy()
        list_=[]    
        ran=random.random()
      
        data1[data1>ran]=1
        data1[data1<ran]=0
        
        
        for i in range(7):
            data_temp=data1.loc[[new_index[i+1],new_index[0]]]
            #print()
            numpy_array = data_temp.to_numpy().T
            #print(numpy_array)
            list_.append(ccpt(numpy_array))
        array_data = np.array(list_)
        #print(array_data)
        res_all+=(array_data)/t
        
    

    res=pd.DataFrame(res_all,columns=['P(B=1 | A=0)','P(B=0 | A=0)','P(B=1 | A=1)','P(B=0 | A=1)'],index=new_index[1:])
    print(res)
   