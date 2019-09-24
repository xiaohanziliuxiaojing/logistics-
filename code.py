# 导入第三方模块
from sklearn import linear_model#调用线性回归
from sklearn.model_selection import train_test_split
from sklearn import metrics #调用衡量指标模块
from sklearn.metrics import classification_report
from sklearn import preprocessing #对数据进行标准化
#from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler#对数据进行标准化
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt



# -----------------------第一步 观察数据特征 ----------------------- #


# 读取数据
data = pd.read_excel(r'E:\小静--工作代码保存\logistic模型\data.xlsx')
#查看变量
data.shape
#检查是否有空值
data.isnull().sum(axis = 0)
data.zhimafen = data.zhimafen.fillna(int(data.zhimafen.mean()))

#计算不同的属性值的个数并绘图
#可以了解数据分布情况，从图形可以看到数据分布及其不平衡
count_uid= data['uid'].groupby(data['is_Losing_uid']).count()
print (count_uid)


# 提取出所有自变量名称
predictors = data.columns[4:-1]
# 构建自变量矩阵
X = data.ix[:,predictors]#选取某几列
#X = preprocessing.scale(X)
X = StandardScaler().fit_transform(X)

# 提取y变量值
y = data.is_Losing_uid

# -----------------------第二步 建模 ----------------------- #

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)
# 利用训练集建模
sklearn_logistic = linear_model.LogisticRegression()
sklearn_logistic.fit(X_train, y_train)
#返回模型的各个参数
print(sklearn_logistic.intercept_, sklearn_logistic.coef_)
# 模型预测
sklearn_predict = sklearn_logistic.predict(X_test)
# 预测结果统计
pd.Series(sklearn_predict).value_counts()



#重抽样前的类别比例
print(y_train.value_counts()/len(y_train))
sklearn_predict = sklearn_logistic.predict(X_test)
#运用smote算法实现训练数据集的平衡
over_samples = SMOTE(random_state = 1234)
over_samples_X,over_samples_y = over_samples.fit_sample(X_train,y_train)
#重抽样后的类别比例
print(pd.Series(over_samples_y).value_counts()/len(over_samples_y))
# 利用训练集建模
sklearn_logistic = linear_model.LogisticRegression()
sklearn_logistic.fit(X_train, y_train)
#返回模型的各个参数
print(sklearn_logistic.intercept_, sklearn_logistic.coef_)
# 模型预测
resample_pred = sklearn_logistic.predict(np.array(X_test))
# 预测结果统计
pd.Series(resample_pred).value_counts()



#预测检验
cm = metrics.confusion_matrix(y_test, sklearn_predict, labels = [0,1])#样本真实分类结果/样本预测分类的结果/所给出的类别
Accuracy = metrics.scorer.accuracy_score(y_test, sklearn_predict)#模型准确率
Sensitivity = metrics.scorer.recall_score(y_test, sklearn_predict)#正例覆盖率
Specificity = metrics.scorer.recall_score(y_test, sklearn_predict, pos_label=0)#负例覆盖率
print('模型准确率为%.2f%%:' %(Accuracy*100))
print('正例覆盖率为%.2f%%' %(Sensitivity*100))
print('负例覆盖率为%.2f%%' %(Specificity*100))
print('每个类别的精确率和召回率:', classification_report(y_test, sklearn_predict))



# -----------------------第三步 预测构建混淆矩阵 ----------------------- #



# 混淆矩阵的可视化
import seaborn as sns
import matplotlib.pyplot as plt
# 绘制热力图
sns.heatmap(cm, annot = True, fmt = '.2e',cmap = 'GnBu')#传入数据
# 图形显示
plt.show()




# -----------------------第四步 绘制ROC曲线 ----------------------- #

# y得分为模型预测正例的概率
y_score = sklearn_logistic.predict_proba(X_test)[:,1]
# 计算不同阈值下，fpr和tpr的组合值，其中fpr表示1-Specificity，tpr表示Sensitivity
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
# 计算AUC的值
roc_auc = metrics.auc(fpr,tpr)

# 绘制面积图
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
# 添加边际线
plt.plot(fpr, tpr, color='black', lw = 1)
# 添加对角线
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
# 添加文本信息
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
# 添加x轴与y轴标签
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
# 显示图形
plt.show()



# -----------------------第五步 此模型用来预测新数据 ----------------------- #

# 读取要预测的数据
data3 = pd.read_excel(r'E:\小静--工作代码保存\logistic模型\yuce.xlsx')
data3.shape
data3.isnull().sum(axis = 0)
data3.zhimafen = data3.zhimafen.fillna(int(data3.zhimafen.mean()))

# 提取出所有自变量名称
predictors = data3.columns[4:]
# 构建自变量矩阵
yuce =data3.ix[:,predictors]#选取某些列


#进行预测
sklearn_predict = sklearn_logistic.predict(yuce)
predict = pd.DataFrame(sklearn_predict,columns=['result'])

# 预测结果统计输出
pd.Series(sklearn_predict).value_counts()
result = pd.concat([data3,predict],axis=1)
writer = pd.ExcelWriter('E:/小静--工作代码保存/logistic模型/result.xlsx')


try:
	result.to_excel(writer,sheet_name='result',index=0)
	writer.save()
	writer.close()

except Exception as err:
	fillte='导出失败:'+str(err)
	print(fillte)
else:
	succefull='导出成功'
	print(succefull)

