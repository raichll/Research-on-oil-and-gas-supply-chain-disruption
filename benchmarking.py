import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
import pandas as pd
# 时间序列数据
new_index = ['Supply chain disruption', 'Technical', 'Economic', 'Social', 'Political', 'Safety', 'Environmental', 'Legal']

data = np.array([1.000000, 0.714036, 0.532305, 0.537565, 0.557059, 0.438277, 0.334784, 0.375230, 0.366774, 0.371231,
                 0.901383, 0.729016, 0.642366, 0.340265, 0.254825, 0.397295, 0.304850, 0.273210, 0.179087, 0.227940,
                 0.467023, 0.616667, 0.584913, 0.518414, 0.498193, 0.342653, 0.348891, 0.415023, 0.433046, 0.687992,
                 0.541386, 0.568399, 0.512923, 0.387029, 0.297888, 0.297118, 0.303473, 0.276949, 0.187940, 0.225487,
                 0.645632, 0.728184, 0.714355, 0.668059, 0.576418, 0.370354, 0.508886, 0.473511, 0.487936, 0.512804,
                 0.468413, 0.491836, 0.399402, 0.268114, 0.297098, 0.271314, 0.219519, 0.199767, 0.126221, 0.181567,
                 0.407256, 0.418602, 0.331566, 0.232485, 0.274974, 0.228435, 0.225713, 0.107361, 0.112562, 0.110412,
                 0.432970, 0.421930, 0.481082, 0.299291, 0.247318, 0.198078, 0.240852, 0.186413, 0.204887, 0.170526])

# 取前10个数据作为测试集，剩余作为训练集
test_size = 10
list_mean_squared_error=[]
list_mean_absolute_error=[]
fig, axs = plt.subplots(2, 4, figsize=(14, 16))
for i in range(8):
    train = data[i*10:test_size+i*10]  # 剩余部分作为训练集
    test = data[i*10:test_size+i*10]   # 前10个作为测试集
    
    # 准备输入数据函数
    def prepare_data(data, history_size):
        X, y = [], []
        for i in range(len(data) - history_size):
            X.append(data[i:i + history_size])
            y.append(data[i + history_size])
        return np.array(X), np.array(y)

    history_size = 1  # 使用过去3个时间点的值来预测下一个值

    # 1. **灰度预测 GM(1,1)**
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

        return fitted


    # GM(1,1)预测
    gm_pred = gm11(test,predict_length=10)

    # 2. **XGBoost 预测**
    X_train, y_train = prepare_data(train, history_size)
    X_test, y_test = prepare_data(test, history_size)

    # XGBoost 模型训练
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    xgb_model.fit(X_train, y_train)

    # 预测
    xgb_pred = xgb_model.predict(np.array(test).reshape(-1,1))

    '''# 3. **ARIMA 预测**
    arima_model = ARIMA(train, order=(1, 1, 1))  # ARIMA(1, 1, 1)
    arima_model_fit = arima_model.fit()

    # 预测
    arima_pred = arima_model_fit.forecast(steps=test_size)'''

    # 4. **线性回归 (LR) 预测**
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # 随机森林预测
    lr_pred = lr_model.predict(np.array(test).reshape(-1,1))

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(np.array(test).reshape(-1,1))
    #支持向量机
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)  # 'rbf' 是径向基函数内核
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(np.array(test).reshape(-1,1))

    # 评估并绘制对比图
    #plt.figure(figsize=(12, 8))
    if i<4:
        
        axs[0,i].plot(range(len(test)), test, label="True Values", color="blue", linestyle='-', marker='o')
        axs[0,i].plot(range(len(gm_pred[:test_size])), gm_pred[:test_size], label="GM(1,1) Predictions", color="green", linestyle="--", marker='x')
        axs[0,i].plot(range(len(xgb_pred)), xgb_pred, label="XGBoost Predictions", color="red", linestyle="--", marker='s')
        #axs[0,i].plot(range(len(arima_pred)), arima_pred, label="ARIMA Predictions", color="purple", linestyle="--", marker='^')
        axs[0,i].plot(range(len(lr_pred)), lr_pred, label="Linear Regression Predictions", color="orange", linestyle="--", marker='d')
        axs[0,i].plot(range(len(rf_pred)), rf_pred, label="Random Forest Predictions", color="grey", linestyle="--", marker='d')
        axs[0,i].plot(range(len(svr_pred)), svr_pred, label="SVR Predictions", color="pink", linestyle="--", marker='d')
        axs[0,i].set_title(f"{new_index[i]}'s series prediction comparison")
    
        axs[0,i].legend()
    else:
        axs[1,i-4].plot(range(len(test)), test, label="True Values", color="blue", linestyle='-', marker='o')
        axs[1,i-4].plot(range(len(gm_pred[:test_size])), gm_pred[:test_size], label="GM(1,1) Predictions", color="green", linestyle="--", marker='x')
        axs[1,i-4].plot(range(len(xgb_pred)), xgb_pred, label="XGBoost Predictions", color="red", linestyle="--", marker='s')
        #axs[1,i-4].plot(range(len(arima_pred)), arima_pred, label="ARIMA Predictions", color="purple", linestyle="--", marker='^')
        axs[1,i-4].plot(range(len(lr_pred)), lr_pred, label="Linear Regression Predictions", color="orange", linestyle="--", marker='d')
        axs[1,i-4].plot(range(len(rf_pred)), rf_pred, label="Random Forest Predictions", color="grey", linestyle="--", marker='d')
        axs[1,i-4].plot(range(len(svr_pred)), svr_pred, label="SVR Predictions", color="pink", linestyle="--", marker='d')
        axs[1,i-4].set_title(f"{new_index[i]}'s series prediction comparison")
        axs[1,i-4].legend()
    list_mean_squared_error.append([mean_squared_error(test, gm_pred[:test_size]),
                                    mean_squared_error(test, xgb_pred),
                                    mean_squared_error(test, lr_pred),
                                    mean_squared_error(test, rf_pred),
                                    mean_squared_error(test, svr_pred)])
    list_mean_absolute_error.append([mean_absolute_error(test, gm_pred[:test_size]),
                                    mean_absolute_error(test, xgb_pred),
                                    mean_absolute_error(test, lr_pred),
                                    mean_absolute_error(test, rf_pred),
                                    mean_absolute_error(test, svr_pred)])


    # 评估各模型
    '''print("XGBoost MSE:", mean_squared_error(test, xgb_pred))
    #print("ARIMA MSE:", mean_squared_error(test, arima_pred))
    print("Linear Regression MSE:", mean_squared_error(test, lr_pred))
    print("GM MSE:", mean_squared_error(test, gm_pred[:test_size]))
    print("Random Forest MSE:", mean_squared_error(test, rf_pred))

    print("XGBoost MAE:", mean_absolute_error(test, xgb_pred))
    #print("ARIMA MAE:", mean_absolute_error(test, arima_pred))
    print("Linear Regression MAE:", mean_absolute_error(test, lr_pred))
    print("GM MAE:", mean_absolute_error(test, gm_pred[:test_size]))
    print("Random Forest MSE:", mean_absolute_error(test, rf_pred))'''

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()

df_MSE = pd.DataFrame(list_mean_squared_error, columns=['GM', 'XGBoost', 'Linear Regression',"Random Fores",'SVR'],index=new_index)
df_MAE=pd.DataFrame(list_mean_absolute_error, columns=['GM', 'XGBoost', 'Linear Regression',"Random Fores",'SVR'],index=new_index)

fig, axs2 = plt.subplots(1, 2, figsize=(14, 16))


df_MSE.plot(kind='bar', ax=axs2[0] , width=0.9, title='Comparative Analysis of Predictive Method Performance: MSE Evaluation' )
axs2[0].set_xlabel('ID')
axs2[0].set_ylabel('MSE_value')

# 绘制第二个 DataFrame 的柱状图
df_MAE.plot(kind='bar', ax=axs2[1],  width=0.9, title='Comparative Analysis of Predictive Method Performance: MAE Evaluation')
axs2[1].set_xlabel('ID')
axs2[1].set_ylabel('MAE_value')

plt.setp(axs2[0].xaxis.get_majorticklabels(), rotation=15)  # 设置第一个子图的 x 轴标签横着放
plt.setp(axs2[1].xaxis.get_majorticklabels(), rotation=15)  # 设置第二个子图的 x 轴标签横着放
plt.setp(axs2[0].yaxis.get_majorticklabels(), rotation=0)  # 设置第一个子图的 x 轴标签横着放
plt.setp(axs2[1].yaxis.get_majorticklabels(), rotation=0)  # 设置第二个子图的 x 轴标签横着放

# 自动调整布局，防止重叠
plt.tight_layout()

# 显示图形
plt.show()