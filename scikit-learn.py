scikint-learn库是Scipy工作集合、的一部分
模型的特征 property
评估算法需要把算法分为训练集和预测集，从前者学数据的特征，
后者测试得到的特征

8.3 用scikint-learn 实现有监督的学习
-----lris 数据集 燕尾花卉数据集 数据来自三种花卉 表示萼片和花瓣的长度
属于内置的数据集

====PCA对象的方法
fit(X,y=None) #每个需要训练的算法否需要fit()方法，是训练步骤
pca.fit(X) #用X训练pca对象

fit_transfrom(X)  #用X来训练PCA模型，同时姜维
newX = pca.fit_transfrom(X) newX就是姜维后的数据

inverse_transform() #将姜维后的数据还远成元数据
X=pca.inverse_transform(newX)

transform()
将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，
都可以用transform方法来降维。

此外，还有get_covariance()、get_precision()、
get_params(deep=True)、score(X, y=None)等方法，



import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import linear_model

diabetes = datasets.load_diabetes()
x_train = diabetes.data[:-20]
y_train = diabetes.target[:-20]
x_test = diabetes.data[-20:]
y_test = diabetes.target[-20:]
plt.figure(figsize=(8,12))
linreg = linear_model.LinearRegression()
for f in range(0,10):
    xi_test = x_test[:,f]
    xi_train = x_train[:,f]
    xi_test = xi_test[:,np.newaxis] #b变二维
    xi_train = xi_train[:,np.newaxis]  
   
    linreg.fit(xi_train,y_train)
    y = linreg.predict(xi_test)
    
    plt.subplot(5,2,f+1)
    plt.scatter(xi_test,y_test,color='k')
    plt.plot(xi_test,y,color='r',linewidth=3)


====支持向量分类
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm

x = np.array([[1,3],[1,2],[1,1.5],[1.5,2],[2,3],
              [2.5,1.5],[2,1],[3,1],[3,2],[3.5,1],[3.5,3]]) #存放了数据点属性值
y = [0]*6 + [1]*5  #存放了数据点的类别信息 6个0 + 5个1 共十一组数据点
svc = svm.SVC(kernel='rbf',C=1,gamma=3).fit(x,y)  #内核 ‘linear’,'poly  degree=3',
#‘rbf’gamma=3
#正则化c 0dao1 表示选取数据点的数量的多少，几泛化能力
X,Y = np.mgrid[0:4:200j,0:4:200j]
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z =Z.reshape(X.shape)
plt.contourf(X,Y,Z > 0,alpha=0.4)  #轮廓线
plt.contour(X,Y,Z,colors=['k','k','k'],linestyles=['--','-','--'],levels=[-1,0,1])
plt.scatter(svc.support_vectors_[:,0],svc.support_vectors_[:,1],s=120,facecolors = 'none')
plt.scatter(x[:,0],x[:,1],c=y,s=50,alpha=0.9)  #c是category类别,颜色序列    矢量


++++++绘制SVM分类器对iris数据集的分类效果

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets,svm

iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target
h = .05
svc = svm.SVC(kernel='linear',C=1.0).fit(x,y)
x_min,x_max = x[:,0].min() - .5,x[:,0].max() + .5
y_min,y_max = x[:,1].min() - .5,x[:,1].max() + .5
h = .02
X,Y = np.meshgrid(np.arange(x_min,x_max),np.arange(y_min,y_max))
Z = svc.predict(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)
plt.contourf(X,Y,Z,alpha=0.4)
plt.contour(X,Y,Z,color='k')
plt.scatter(x[:,0],x[:,1],c=y)

----非线性内核，多项式内核










带有误差线的条状图=====
std = [0.8,1,0.4,0.9,1.2]
plt.barh(index,values,xerr=std,error_kw={'ecolor':'0.1','capsize':6},  #宽度
        alpha=0.9,label='First')
plt.yticks(index,['A','B','C','D','E'])
plt.legend(loc=5)

直方图
plt.hist()===
多序列条状图
plt.bar()===
水平条状图
plt.barh()===

7.13.3 位pandas DataFrame生成多序列条状图===
df.plot(kind='bar')  #df位DataFrame对象

多序列堆积条状图===
plt.bar(index,series1,color='r')
plt.bar(index,series2,color='b',bottom=series1)
plt.bar(index,series3,color='y',bottom=(series1+series2))
series 是ndarray对象

水平堆积条状图===
plt.barh(index,series1,color='r')
plt.barh(index,series2,color='b',left=series1)
plt.barh(index,series3,color='y',left=(series1+series2))
series 是ndarray对象

以不同的影线条虫条状图===
plt.barh(index,series1,color='w',hatch='xx')  #孵化
plt.barh(index,series2,color='w',hatch='///',left=series1)
plt.barh(index,series3,color='w',hatch='\\\\\\',left=(series1+series2))
series 是ndarray对象


===为pandas DatdFrame绘制堆积条状图
df.plot(kind='bar',stacked=True)   #堆叠的


条状图表现对比关系
===plt.ylim(-7,7)  #取到-6到6
plt.bar(x0,y1,0.9,facecolor='r',edgecolor='w')  #边缘
plt.bar(x0,-y2,0.9,facecolor='b',edgecolor='w')   #0.9是间距
====== for x,y in zip(xo,y1):
			plt.text(x + 0.4,y + 0.05,'%d' % y,ha='center',va='bottom')  #ha水平对齐 ，va垂直对齐
for x,y in zip(x0,y2):
	plt.text(x + 0.4,-y - 0.05,'%d' % y,ha='center',va='top')


7.14 饼图
labels = ['one','two','three','four0']
values = [10,30,45,15]
colors = ['yellow','green','red','blue']
pxplode = [0.3,0,0,0]
plt.pie(values,labels=labels,colors=colors,
	explode=explode,startangle=180，
	autopct='%1.1f%%',shadow=True)====
#explode 设置脱离出来 取值0到1
#startangle 设置旋转角度 取值0到360
#autopct 设置显示百分比
#shadow 设置影印
plt.axis('equal')=====

为DataFrame绘制饼图
df['series1'].plot(kind='pie',figsize=(6,6))===




=====高级图表


等值线图=====
import matplotlib.pyplot as  plt
import numpy as np
dx = 0.01;dy = 0.01 # x和y的步长
x = np.arange(-2.0,2.0,dx) #确定x的取值范围
y = np.arange(-2.0,2.0,dy) 
X,Y = np.meshgrid(x,y) #用于生成网格采样点的函数。
def f(x,y):#计算z的值
    return (1 - y**5 + x**5)*np.exp(-x**2-y**2)

C = plt.contour(X,Y,f(X,Y),8,colors='black') #轮廓，8轮廓条线，颜色是黑色
plt.contourf(X,Y,f(X,Y),8) #轮廓线
plt.clabel(C,inline=1,fontsize=10) ##轮廓标签，内联的，大小10
plt.colorbar()   #颜色柱状

优化后的图===
plt.contourf(X,Y,f(X,Y),8,cmap=plt.cm.hot)


====及区图
N = 8   #分8部分
theta = np.arange(0.,2 * np.pi, 2* np.pi / N)  #角度
radii = np.array([4,7,5,3,1,5,6,7])  #半径
plt.axes([0.025,0.025,0.95,0.95],polar=True)   #重点，不写是条状图
colors = np.array(['red','green','yellow',
	'#4bb2c5','blue','black','#839577','#EAA228'])
bars = plt.bar(theta,radii,width=(2*np.pi/N),
	bottom=0.0,color=colors)


====3D图像===

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()   #构架一个figure对象
ax = Axes3D(fig)  #以fogur对象为参数，构建axes对象
x = np.arange(-2,2,0.1)
y = np.arange(-2,2,0.1)
X,Y = np.meshgrid(x,y)  #网格
def f(x,y):
    return (1 - y**5 + x**5)*np.exp(-x**2-y**2)
ax.plot_surface(X,Y,f(X,Y),rstride=1,cstride=1)  #步幅 控制格子的大小
函数要注意



===优化后
ax.plot_surface(X,Y,f(X,Y),rstride=1,cstride=1,cmap=plt.cm.hot)  #步幅
ax.view_init(elev=30,azim=125)
cmap参数修改颜色，颜色表     ax.view_init()修改颜色，第一个参数
指定从哪个角度查看曲面，第二个指定曲面旋转的角度


3D散点图====

xyz的真谛有随机数指定
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x2,y2,z2,c='r',marker='^')  #分散，做记号
ax.set_xlabel('X Label')

3D条状图=====
import  matplotlib.pyplot as plt
import numpy as  np
from mpl_toolkits.mplot3d import Axes3D
x  = np.arange(8)
y = np.random.randint(0,10,8)
y2 = y + np.random.randint(0,3,8)
y3 = y2 + np.random.randint(0,3,8)
y4 = y3 + np.random.randint(0,3,8)
y5 = y4 + np.random.randint(0,3,8)    #y其实是z的作用
clr = ['#4bb2c5','#c5b47f','#EAA228','#579557','#839557','#958c12','#953579','#4b5de4']
fig = plt.figure()=====
ax = Axes3D(fig)======
ax.bar(x,y,0,zdir='y',color=clr)
ax.bar(x,y2,10,zdir='y',color=clr) #zdir是图像起来,垂直于y轴
ax.bar(x,y3,20,zdir='y',color=clr)  注意函数
ax.bar(x,y4,30,zdir='y',color=clr)
ax.bar(x,y5,40,zdir='y',color=clr)
ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')
ax.view_init(elev=45)


多面版图形=====
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8]) 
 #前两个数表示，左下角的位置，后两个表示长
inner_ax = fig.add_axes([0.6,0.6,0.25,0.25])  #左底宽高
x1 = np.arange(10)
y1 = np.array([1,2,7,1,5,2,4,2,3,1])
x2 = np.arange(10)
y2 = np.array([1,3,4,5,4,5,2,6,4,3])

ax.plot(x1,y1)
inner_ax.plot(x2,y2)
注意函数



子图网格=====注意函数
gs = plt.GridSpec(3,3)  #把绘图区域分成多个子区域  9个 网格+规格
fig = plt.figure(figsize=(6,6))  #指定大小
x1 = np.array([1,3,2,5])
y1 = np.array([4,3,7,2])
x2 = np.arange(5)
y2 = np.array([3,2,4,6,4])

s1 = fig.add_subplot(gs[1,:2])   #次要情节
s1.plot(x1,y1,'r')

s2 = fig.add_subplot(gs[0,:2])
s2.bar(x2,y2)

s3 = fig.add_subplot(gs[2,0])
s3.barh(x2,y2,color='g')

s4= fig.add_subplot(gs[:2,2])
s4.plot(x2,y2,'k')

s5 = fig.add_subplot(gs[2,1:])
s5.plot(x1,y1,'b^',x2,y2,'yo')
