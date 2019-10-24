numpy   的核心是ndarray对象==及数组==，由====同质元素=====组成的n维数组，即类型和大小相同
#数据类型有dtype的numpy对象指定，，数组的维数和元素数由数组的型shape()来指定

#定义ndarray用array()函数，以python列表作为参数 ，列表的元素
#为ndarray的元素

1，numpy 内置了并行运算的功能，当系统有多个核心时，
做魔种计算时，会自动进行并行运算
2，底层是c语言写的，内部解除了GIL（全局解释所），对数组的操作不受
python解释器的影响
3，有一个强大的n维数组对象Array
4，实用的线性代数，傅里叶变换和随机数生成函数
是一个非常高效的用于处理数值型运算的包

import numpy as np
a =np.array([1,1,2])
print(type(a))
#数组的秩
---a.ndim
1
#数组的长度
---a.size
3
#数组的型 
---a.shape
(3l,)

#二维数组
b = np.array([1,1],[2,2])
#是真的数据类型
b.dtype
b.ndim
2
b.size
4
b.shape
(2l,2l)

---
b.resize((1 ,4))  修改原数组

b.flatten() 拷贝，不会改变原数组

b.ravel() 返回视图，会改变原数组
b.flat


#这个数组有两条轴，所以秩为2，每条轴的长度为2 

#ndarraay对象的属性之一 itemsize,定义每个数组元素的长度为几个字节 
---b.itemsize
8

=====函数生成特殊数组=====
a1= np.zeros((2,2))
a2 = np.ones((3,2))
a3 = np.full((3,2),8)  #全是8
a4 = np.eye(3)  #斜方型上全是1，其他的都是0  3X3



------------#创建数组的几种方法如下-：
1,array()函数，参数为单层或嵌套列表或元组
2,zeros()函数,元素为0
np.zeros((3,3))
3，ones()函数,元素为1
np.ones((3,3))
两个函数默认使用float64数据类型创建数组
4，numpy的 arange()函数，很有用,类似函数 linspace()
np.arange(0,10)
一维数组 0-9
np.arange(0,12,3)
array([0,3,6,9]) 以3等间隔，三个参数都可以是浮点型
np.linspace(0,10,5) 第三个参数为分成几部分 5
array([0.,2.5,5.,7.5,10.])
5,使用随机数填充数组  numpy.random模块的 random()函数
 数量由参数指定
np.random.random(3) 还可以二维 （(3,3)）
np.raandom.randint()
pn.random.randn()  从“标准正态”分布中返回一个样本

----#生成二维数组
np.arange(0,12).reshape((3,4))  ((12,)) 一维 
#reshape中有了几个元素就代表几维 （例子2维）

#数据类型
1.字符串类型
g = np.array(['a','b','c'])
g.dtype
'Sl'
g.dtype.name
'String8'

还有很多的数据类型

#dtype选项 
#复数数组
f = np.array([[1,2,3],[4,5,6],dtype=complex])

----基本操作
+ - * 元素级运算，数与数组或数组和数组

自增，自减
a += 1   a -= 1

矩阵积
------在其他数据分析工具中 * 算矩阵积，行*列+
numpy用函数 dot() 不是元素级的
np.dot(a,b) a,b为数组


-----链接数组，使用栈的概念
一，两个数组之间
1，垂直生长函数  np.vstack((A,B)) 垂直入栈
2, 水平生长函数   np.hstack((A,B))  水平入栈
3，concatenate([A,B],axis=1) axis=None 数组扁平化，为一维数组


二，多个数组之间  一维数组
1,垂直 np.column_stack((a,b,c))
2,水平 np.row_stack((a,b,c))

----数组切分
水平切分函数 hsplit()
垂直切分函数 vsplit()
[B,C] = np.hsplit(A,2) 分两部分，可以指定切割位置(A,(1,3))
[B,C] = np.vsplit(A,2) 分两部分

split()函数
[a,b,c] = np.split(A,[1,3],axis=1)
[1,3]为列索引
[a,b,c] = np.split(A,[1,3],axis=0)
[1,3]为行索引

---为原数组生成副本
函数 copy()
c = a.copy() c为a的副本

---向量化和广播机制

---数组的读写
1，以二进制读写数据
函数 save()和load()
np.save('save_date',a) a为numpy数组  .npy扩展会自动添加
np.load('save_date.npy') a为numpy数组 .npy要一定写上

操作CSV文件 =====
读取csv文件  
np.savetxt('data.csv',frame,delimiter=','，header='数学，英语'，comments='')
np.loadtxt()  只能操作二维和一维数组
help(np.loadtxt)

=====写入csv文件
def write_csv1():
    import csv
    headers = ['id', 'name', 'age', 'height']
    values = [(12, 'r', 34, 126),
              (12, 'r', 34, 126),
              (12, 'r', 34, 126)]
    with open('demo.csv', 'w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        writer.writerows(values)

def write_csv2():
    import csv
    headers = ['id', 'name', 'age', 'height']
    values = [{'id':12,'name':'rbp','age':34,'height':1 },
              {'id':12,'name':'wws','age':56,'height':3 },
              {'id':12,'name':'ln','age':23,'height':2 }
              ]
    with open('demo2.csv','w',encoding='utf-8',newline='') as fp:
        writer = csv.DictWriter(fp,headers)
        writer.writeheader()
        writer.writerows(values)

if __name__ == '__main__':
    write_csv1()
    write_csv2()


-------2.读取文件中的列表形式数据------
函数 genfromtxt() 读取文本文件中的数据并将其插入到数组中 
优点是能处理缺失数据
data = np.genfromtxt('data.csv',delimiter=',',names=True) 分隔符
按列取元素  data['id']   按行  data[0]


-----基本函数
a为numpy数组
#正弦值
np.sin(a)运算级
#平方根
np.sqrt(a)元素级运算

np.any(a==1) 布尔判断
(a>=0).all()
np.take(index,a) 从指定的小标区指定的值




-====拷贝
1、不拷贝   a is b  True 
2.浅拷贝  c = a.view() a is c     False  拷贝指针
3.深拷贝   d = a.copy()  a is d False  还拷贝数据

数据清洗-====
df.dropna(how='all')   axis=1  #删除有空的所有行，删除列
df.dropna(thresh=2)  #删除缺失值有两个的行

填充数据====
df.fillna(0)  #趋势值填充常数，修改原来数据
df.fillna({1:0.9,2:0}，inplace=True)  #把所有为1的列填充为0.9，位2的填充0
不改变原来的值,为TRUE时改变原来的值

===df.fillna(method='ffill')  #把缺失值填充为上一个数，不断向下赋值

数据转换====
检查重复的数据行，返回布尔值
data.duplicated()  #复制的，复制品
删除重复行
data.drop_duplicates()  返回DataFrame对象
data.drop_duplicates(['key'])  #按列删除行
data.drop_duplicates(['k1','k2'],keep='last')  留下最后一组重复数据


low = data['food'].str.lower()   #转换成小写


NAN和INF的处理======

nan是浮点型 无穷大
1，删除
import numpy as np
data = np.random.randint(0,10,size=(3,3)).astype(np.float)
data[2,2],data[1,1] = np.nan,np.nan
# data[~np.isnan(data)]

lines = np.where(np.isnan(data))[0]   
#返回时nan的行和列的index  元组 取行
np.delete(data,lines,axis=0)
删除缺失值所在行


np.random.seed(1) === 种子

np.random.rand(2,3) 0dao1的随机数
np.random.randn()  根据标准正太分布的值

np.random.randint(0,10，size=(10,))
np.random.choice(data,3)  随机选择3个
np.random.choice(5,size=(3,4))  
np.random.shuffle(data) 随机打乱
np.random.uniform(x,y) 方法将随机生成下一个实数，它在[x,y]范围内。
numpy.random.normal(loc=0,scale=1e-2,size=shape)
# 参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
# 参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
# 参数size(int 或者整数元组)：输出的值赋在shape里，默认为None。

#随机打乱数据，长用在分离训练集和测试集
np.random.permutation(num)  #随机生成固定个数
numpy.random.shuffle()

np.r_：是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
类似于pandas中的concat()
np.c_：是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，
类似于pandas中的merge()

np.newaxis 为 numpy.ndarray（多维数组）增加一个轴
x = x.reshape(-1,1)
plt.plot(np.sort(x),y2[np.argsort(x)],c='r',linewidth=2)


