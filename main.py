import pandas as pd
import jieba
import re
from gensim.models import Word2Vec
import os
import numpy as np

# 加载停用词表
def load_stopwords(file_paths):
    """
    加载多个停用词表
    :param file_paths: 停用词表文件路径列表
    :return: 停用词的集合
    """
    stopwords = set()
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                stopwords.update(line.strip() for line in f)
        else:
            print(f"停用词表文件 {file_path} 不存在!")
    return stopwords


# 清理非中文字符和停用词
def clean_text(text, stopwords):
    """
    清理文本中的非中文字符和停用词
    :param text: 输入文本
    :param stopwords: 停用词集合
    :return: 清理后的文本
    """
    if not isinstance(text, str):
        return []

    # 使用 jieba 分词
    words = jieba.cut(text)

    # 过滤非中文字符和停用词
    cleaned_words = [word for word in words if word.isalpha() and word not in stopwords]
    return cleaned_words


# 读取 Excel 文件
def read_excel(file_path, sheet_name=0):
    """
    读取 Excel 文件，并返回指定工作表中的数据。
    :param file_path: Excel 文件路径
    :param sheet_name: 工作表名称或索引，默认读取第一个工作表
    :return: DataFrame 对象
    """
    return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')


# 训练 Word2Vec 模型
def train_word2vec(corpus, vector_size=100, window=5, min_count=2, workers=4):
    """
    训练 Word2Vec 模型。
    :param corpus: 分词后的语料
    :param vector_size: 向量维度
    :param window: 上下文窗口大小
    :param min_count: 最低词频
    :param workers: 并行处理的线程数
    :return: 训练好的 Word2Vec 模型
    """
    return Word2Vec(sentences=corpus, vector_size=vector_size, window=window,
                    min_count=min_count, workers=workers, sg=0)  # CBOW 模式


# 保存和加载模型
def save_model(model, file_path):
    """
    保存 Word2Vec 模型。
    :param model: Word2Vec 模型
    :param file_path: 模型保存路径
    """
    model.save(file_path)


def load_model(file_path):
    """
    加载 Word2Vec 模型。
    :param file_path: 模型文件路径
    :return: 加载的 Word2Vec 模型
    """
    return Word2Vec.load(file_path)


# 主执行代码
if __name__ == "__main__":
    # 指定文件路径
    file_path = "data.xlsx"
    stopword_files = ["zh_stopwords.txt", "baidu_stopwords.txt",'cn_stopwords.txt'
                      ,'hit_stopwords.txt','scu_stopwords.txt']
    seed_word_total = [
        ["供应链", "供应商", "供应", "供需", "上游", "下游", "采购", "成本", "原材料", "价格上涨", "减产", "紧缺", "周转", "储备", "故障", "经营风险", "中断",
         "停工", "运费", "运力"],
        ["故障", "维修", "开采", "调度", "延迟", "效率", "风险", "成本", "调度", "时间", "系统", "能力", "发展", "运行", "减少", "科技", "学习", "计算",
         "工程", "前沿"],
        ["经济", "影响", "股票", "市场", "行业", "需求", "财务", "风险", "金融", "混乱", "商品", "萧条", "衰退", "减少", "下降", "随机", "负面", "波动",
         "杠杆", "流动性"],
        ["社会", "工资", "安全", "健康", "薪酬", "政府", "维护", "地位", "规范", "界限", "人口", "障碍", "负面", "退化", "缺失", "延误", "压力", "焦虑",
         "挑战", "损失"],
        ["政府", "规章", "制度", "政治", "不利", "问题", "持续", "糟糕", "降低", "危机", "干预", "争端", "不当", "负面", "交付", "停顿", "抗议", "混乱",
         "组织", "处理"],
        ["安全", "中断", "危害", "损失", "修复", "伤害", "阻碍", "干扰", "受伤", "评估", "死亡", "措施", "改进", "有害物", "意外", "停工", "死亡率", "供应链",
         "事故", "加剧"],
        ["环境", "污染物", "有害成分", "污染", "自然灾害", "人文灾害", "中断", "地震", "火灾", "暴雨", "变形", "损害", "堵塞", "封锁", "泄露", "废物", "成本增加",
         "延误", "破坏", "意外"],
        ["法律", "程序", "政治", "规避", "暴露", "负面", "焦虑", "管辖", "损失", "条款", "协定", "中断", "权利", "制约", "后果", "挑战", "条件", "问题",
         "阻碍", "影响"]
    ]
    # 加载停用词表
    stopwords = load_stopwords(stopword_files)

    # 读取数据
    df = read_excel(file_path)

    top_n = 200
    final_top_n = 70
    similarity_ratio = 0.6


    # 假设需要清理的列名为 'content'
    if 'content' in df.columns:
        df['cleaned_result'] = df['content'].apply(lambda x: clean_text(x, stopwords))

    # 保存清理后的结果
    df.to_excel("cleaned_result.xlsx", index=False)
    # 提取清理后的语料
    #corpus = df['cleaned_result'].tolist()
    #corpus = df['date_year','cleaned_result'].where(df['date_year']==2014).tolist()
    #print(corpus)
    list_=[]
    for i in range(8):
        list_temp=[]
        for j in range(10):
            corpus = df.where(df['date_year']==2014+j).dropna(axis=0)['cleaned_result'].to_list()
            # 训练 Word2Vec 模型

            model = train_word2vec(corpus, vector_size=500, window=10, min_count=2, workers=4)

            # 保存模型
            model_file = "word_train.model"
            save_model(model, model_file)

            # 加载训练好的模型
            model = load_model(model_file)

            # 获取与 "管理" 相关的相似词

            '''for seed_word in seed_word_total[i]:
                try:
                    # 获取与 seed_word 相似的 top_n 个词
                    similar_words = model.wv.most_similar(seed_word, topn=top_n)
                    
                except KeyError:
                    pass'''
            list_sc=[]                   
            for seed_word in seed_word_total[i]:
                if seed_word not in model.wv.key_to_index:
                    list_sc.append(seed_word)
            print(list_sc)
            for element in list_sc:
                if element in seed_word_total[i]:
                    seed_word_total[i].remove(element)        
            similar_words = model.wv.most_similar(seed_word_total[i], topn=top_n)
                
            print(similar_words)
            # 过滤相似度大于 0.6 的词
            filtered_words = [word for word, similarity in similar_words if similarity > similarity_ratio]

            # 获取前 final_top_n 个关键词
            final_keywords = filtered_words[:final_top_n]
            print("最终的关键词:", final_keywords)

            # 计算总词数
            total_words = sum(len(sentence) for sentence in corpus)
            #print(f"总词数: {total_words}")

            # 计算关键词的词频
            keyword_frequencies = {keyword: sum(sentence.count(keyword) for sentence in corpus) for keyword in final_keywords}
            total_keyword_count = sum(keyword_frequencies.values())
            #print(f'关键词数:{total_keyword_count}')
            '''# 输出关键词的词频
            print("\n关键词的词频:")
            for keyword, freq in keyword_frequencies.items():
                print(f"{keyword}: {freq}")'''

            # 计算总词数与关键词词频的比例
            #keyword_ratios = {keyword: (freq / total_words) for keyword, freq in keyword_frequencies.items()}
            total_keyword_ratios = total_keyword_count / total_words

            # 输出关键词与总词数的比例

            '''for keyword, ratio in keyword_ratios.items():
                print(f"{keyword}: {ratio:.4f}")'''
            print(f'第{i+1}因子,第{j+1}年的关键词数:{total_keyword_count}')
            print(f"第{i+1}因子,第{j+1}年的总词数: {total_words}")
            print(f"第{i+1}因子,第{j+1}年的总关键词比例: {total_keyword_ratios:.4f}")
            list_temp.append(total_keyword_ratios)
        list_.append(list_temp)
    array_2d = np.array(list_)
    print(array_2d)
    result = pd.DataFrame(array_2d)

    # 将 DataFrame 保存为 Excel 文件
    result.to_excel('result.xlsx', index=False, header=False)

