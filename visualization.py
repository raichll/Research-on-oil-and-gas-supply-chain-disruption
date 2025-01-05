import pandas as pd
import jieba
import re
from gensim.models import Word2Vec
import os
import numpy as np
import matplotlib.pyplot as plt

def preconditioning(df):
    
    df.loc[-1]=df.columns

    df.columns=[f'{i+2014}' for i in range(10)]
    df.sort_index(inplace=True) 
    new_cow =[ '2014' , '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

    new_index = ['Supply chain disruption', 'Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']

    # 将新的索引设置为 DataFrame 的索引
    df.index = new_index
    
    df=df*100
    df[df>1]=0.99
    return df

def Line_chart(df):
        # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图表的大小
    # 绘制每一行的数据（每一行对应折线图中的一条线）
    for row in df.index:
        plt.plot(df.columns, df.loc[row], marker='o', label=row)  # 绘制每一行数据

    # 添加标题
    #plt.title('', fontsize=16)

    # 添加X轴和Y轴标签
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Factor occurrence probability', fontsize=12)

    # 显示网格
    plt.grid(True)

    # 显示图例
    plt.legend(title='Supply chain disruption and its impact factors', fontsize=10)

    # 调整布局，防止标签重叠
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == "__main__":

    df = pd.read_excel('result.xlsx')
    df=preconditioning(df)
    print(df)
    Line_chart(df)



