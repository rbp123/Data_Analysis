1,pandas 的核心数据结构 Series 存储一维数据
DataFrame 存储多维数据

series 由两个数组组成 主数据数组和index索引数组
import pandas as pd
s = pd.Series([12,-4,7,9])
0  12
1  -4
2  7
3  9
dtype:int64
s = pd.Series([12,-4,7,9],index=['a','b','c','d'])
,1，查看两个数组
s.values
s.index
2，选择内部元素
s[2] s['b'] 
7     -4
s[0:2]
a  12
b  -4

s[['b','c']]
b  -4
c  7
3，---给元素赋值
s[2] = 1
s['b'] =1

4，用numpy数组或现有的series对象定义新的series对象,
只是引用，不改变始值
arr = np.array([1,2,3,4])
s3 = pd.Series(arr)
5，筛选元素
s[s > 8]
6,series对象运算和数学函数
+ - * /
s / 2
np.log(s)

-------7,series对象的组成元素
s.unique() 返回去重后的元素
s.value_counts() 返回去重后的元素，斌计算出现的次数
s.isin()  用于筛选数据，返回布尔值
s.isin([0,3])
s[s.isin([0,3])]

8,NaN 非数值 为数组中元素缺失项驶入np.NaN
s2 = pd.Series([5,-3,np.NaN,14])
函数 s.isnull()和s.notnull()   返回布尔值

------9，series用作字典，把series对象当做字典对象
mydict = {'red':100,'blue':200}
myseriees = pd.Series(mydict)

red  100
blue  200

10,Series对象之间的运算
myseries1 + series2  只对标签相同的元素求和，不同的也添加，但值为NaN


------DataFrame对象
有两个索引数组 行标签和列标签

-------1，定义 DataFrame对象
给构造函数 DataFrame() 传递一个dict对象，以每一列的名称作为建
每个键都有一个数组作为值
prince
frame = pd.DataFrame(data)

    color object prince
 0   blue   ball  1.1
 1   green  pen   1.0
 ----选择自己想要的列
frame2 = pd.DataFrame(data,columns=['object','prince'])
相当于求片，取部分

---index行索引默认从0开始，也可自己制定
frame = pd.DataFrame(data,index=['one','two','three'])

----不用dict对象，用函数，参数为 数据矩阵，index选项和columns选项
frame3 = pd.DataFrame(np.arange(16).reshape((4,4)),
	index=['red','blue','yellow','whitle'],
	columns=['ball','pen','pencil','paper']
	)

----2,选取元素
DataFrame对象所有列和索引的名称  frame.columns  frame.index
-----取所有值
frame.values

选取一列的值，只需要把列名传入即可
1，frame['price']  返回series对象,
2，把列名作为DataFrame实例的属性
frame.price

航取值，用ix属性和索引值 返回series对象
frame.ix[2]
用一个数组索引值就可以选取多行
frame.ix[[2,4]]  选两行
还可以用
frame[0:3] 选三行

如果只选取一个元素，需要制定列名称和行索引
frame['object'][3]

4,---赋值
 给行列取名   用name属性
 frame.index.name = 'id'  frame.columns.name = 'item'

 --添加一列新元素
 frame['new'] = [12,12,32,20]
 更新一列 先np.arange() 创建一个序列用于更新
 ser = pd.Series(np.arange(5))
 frame['new'] = ser 
更新一个元素
frame['price'][2] = 3.3

4,元素的所属关系
判断一组元素是否属于DdtaFrame对象 frame.isin() 返回布尔值DataFrame对象
frame.isin([1.0,'pen'])
将上面返回作为条件，得到一个新 DataFrame对象,其中只包含满足条件的数据
frame[frame.isin([1.0,'pen'])]

5,删除一列
删除一整列
del frame['new']

6,筛选
frame[frame < 12]

-----7,用嵌套字典生成 DataFrame对象
nestdict = {

	'red':{2011:22,2013:33},
	'whitle':{2011:13,2012:22,2013:16},
	'blue':{2011:17,2012:27,2013:18}
}
frame = pd.DataFrame(nestdict)
     blue  red  whitle
2011  17    NaN   13
2012  27    22    22
2013  18    33    16

8, DataFrame对象 转置
frame.T


----4,5.3 index对象
1，index对象的方法
返回索引值最小和最大的元素
ser.idmin()  ser.idmax()
调用pandas的index对象的is_unique,判断是否含有重复的索引值
sed.index.is_unique   返回一个布尔值

-----4.6 索引对象的其他功能  ser为Series对象
更换索引
用pandas的 reindex()函数 
ser.reindex(['tree','four','five'])
自动填充
ser.reindex(range(6),method='ffill') range(6)指定范围
DataFrame 索引的更换
frame.reindex(range(6),method='ffill',columns=['colors','price','new'])

删除   
ser.drop('yellow') 返回布包含已删除的索引元素的新对象
删除多个 ser.drop(['yellow','red','blue'])
删除 DataFrame对象的元素需要制定两个轴的轴标签
删除行 fram.drop(['blue','yellow'])
删除列  需制定列名和axis选项制定那个轴删除
frame.drop(['pen','pencil'],axis=1)

------算数和数据对齐
s1和s2 都用的相加，没有的用NaN填充 s1,s2为series对象
s1 + s2
DataFrame对象自己运算，行和列都要对齐

------4.7 数据结构之间的运算
相加
frame1.add(frame2)
sub() div()  mul()

4.7.2 DataFrame对象和 Series对象之间的运算
series的索引和DataFrame的列名一致,不一样会用NaN填充
frame - series

------函数营业映射
通用函数
---1，np.sqrt(frame)  每个元素计算平方根

4.8.2 按行或列执行操作的函数
1，自定义函数，这些还是对一位数组进行运算，返回结果为一个数值
------例如lambda函数
f = lambda x:x.max() - x.min()  计算数组元素取值范围
2,还可以用
def f(x):
	return x.max() - x.min()

------------用 apply()函数可在 DataFrame对象上调用刚定义的函数
返回一个标量
frame.apply(f)  列计算
frame.apply(f，axis=1)  行计算

还可以返回一个Series对象，因而可以借助它执行多个函数
def F(x):
	return pd.Series([x.max(),x.min()],index=['min','max'])

-------4.8.3  统计函数
大多数统计函数对 DataFrame对象依旧有效，因此没必要使用 apply()函数
frame.sum()
frame.mean()

-----describle()函数可以同时计算多个统计量   
frame.describle()
计算出了 mean count std min max 25% 50% 75%

----4.9 排序和排位次
使用索引机制
sort_index()函数返回一个对 ----index -----排序后的新对象
Series对象的排序
ser.sort_index() 默认按升序，参数 ascending=False按降序排列
ser.sort_index(ascending=False)

DataFrame对象的排序
按行排序  对index排序  frame.sort_index()
按列排序  对数据排序    frame.sort_index(axis=1)

----对元素排序
对 Series对象的排序
ser.order()  不能用
对 DataFrame对象的排序
frame.sort_index(by=['pen','pencil']) 多列排序

排位次 为序列的每个元素安排一个位次，初始1，依次加1，位次月靠前，数值越小
ser.rank()  根据值得大小，method=‘first’
默认按升序，参数改变

-------相关性和协方差--------
函数 corr()和 cov()函数  covariance
这两个量的计算通常涉及两个　Series对象
seq1.corr(seq2)
seq1.cov(seq2)

另一种情况是，计算单个 DataFrame对象的相关性和协方差，返回两个新
DataFrame对象形式的矩阵
frame.corr()
frame.cov()

函数 corrwith() 方法计算 DataFrame对象的列或行与 Series对象或其他
DataFrame对象元素两两之间的相关性
frame.corrwith(ser)
frame1.corrwith(frame2)

4.11 NaN数据
4.11. 位元素NaN赋值
用np.Nan 或np.nan 
ser = pd.series([0,1,2,np.nan,9],index=['red','blue','yellow','green'])
或 ser['white'] = None

----------4.11.2 过滤NaN
删除NaN  ser.dropna()
另一种 ser[ser.notnull()]

DataFrame对象处理nan,用 dropna()会删有 nan的所有行和所用列
------只删除所有元素均为nan的航和列   需指定 how传参数----
frame.dropna(how='all')

4.11.3 nan填充其他值------ 用其他元素替代nan/0
frame.fillna({'ball':1,'pen':99})
把0替换掉了


------4.12 等级索引和分级
mser = pd.Series(np.random.random(8),index=[['whitle','whitle',
	'whitle','lue','blue','red','red','red'],['up','down',
	'right','up','down','up','down','left']])
mser.index
选取第一列索引中的莫一索引项的元素
mser['whitle']   
或者 选取第二列索引中某一索引项的元素
mser[:,'up']
选取特定的元素  mser['whitle','up']

----函数 unstack() 把二级索引转换成 DataFrame对象 第二级索引成列
mser.unstack()
若要逆运算   frame.stack() 

对于 DataFrame对象可以指定航和列的等级索
mframe = pd DataFrame(np.random.random(16),index=[
	['whitle','whitle','red','red'],['up','down','up','down']]
	columns=[['pen','pen','paper','paper'],[1,2,1,2]])

-----4.12.1 重新调整顺序和为层级排序
swaplevel()函数以要调整的两个层级的名称作为参数，
返回交换后的一个新对象，其中
各元素的顺序保持不变
mframe.columns.names = ['object','id']
mframe.index.names = ['color','status']

mframe.swaplevel('color','status') 层级互换

sortlevel 函数只根据一个层级对---数据排序  不可用
mframe.sortlevel('colors')

4.12.2 按层级统计数据
DataFrame对象和Series对象的很多描述性和概括统计量都有level选项
可用它指定那个层级的描述性和概括统计量

对行一层级进行统计，把层级的名称赋给level选项即可
mframe.sum(level='colors')
对层级的列进行统计
mframe.sum(axis=1)

层级索引的交换
s.swaplevel() 只交换不排序
s.sortlevel() 交换并排序，先排序外层再内层

s.sum(axis=1,skipna=True) 排除空值
s.idmax() 

frame.set_index(['a','c']) #将Dataframe对象转换成层级索引
frame.reindex()   #与上相反



----深入pandas------
数组的拼接
concatenate([s1,s2],axis=1)


merge() 默认是内连接
对象都有相同的咧
合并  pandas.merge(frame1,frame2,on='id') on属性指定以那一列进行合并
要合并多个键，就把对个键传给 on=['id','brand']

列的名字不同
pandas.merge(frame1,frME2,left_on='id',right_on='sid') 
 -----how属性指定链接方式 取值有 outer left right



---根据索引合并  将ringt_index和left_index的值改为True
pd.merge(fr1,fr2,right_index=True,left_index=True)
-----frame对象的 join()函数跟适合做 索引合并
fr1.join(fr2)  按照索引，列名不能一样====重点

----拼接  函数 concatenate()   ndarray 对象
pd.concatenate([array1,array2],axis=1) 列拼接

----按轴拼接  series和DataFrame对象
pd.concat([ser1,ser2])  默认axis=0    默认外链接，默认过滤缺失数据
属性join 的参数改变链接方式
pd.concat([ser1,ser2]，axis=1,join='inner') 内连接
 keys属性在拼接的轴上创建等级索引
 pd.concat([ser1,ser2],asix=1,keys=[1,2])   设置ser1和ser2数据名称

 6.2.1 组合 我们无法通过合并和拼接组合数据，例如，两个数据集的索引
 完全或部分重合
 combine_first() 函数可以用组合series对象，同时对其数据

 ser1.combine_first(ser2) 按ser1对其数据
 部分合并
 ser[1:3].combine_first(ser2[:3])

 

6.2.2 轴向旋转-----了解
入栈（stacking） 旋转数据结构，把列转换成行
处栈（unstacking）：把行转换成列

frame.stack() 巴列转换成行 生成等级索引 变为Series对象
ser.unstack() 把行转换成列，变为DataFrame对象
为stack传入层级的编号会名称，即可对相应的层级进行操作

可以用层级索引号或层级名===设置 例如frame.unstack(1、姓名)


2.----长格式向宽格式旋转--------
wideframe = longframe.pivot('color','item')

6.2.3 删除
del frame['ball']  删除一列
frame.drop('whitle') 删除一行

6.3 ----- 数据转换
6.3.1 删除重复元素 
检测重复元素，返回布尔值Series对象   函数 duplicated() 复制的
    drop_duplicated()   返回删除后的对象

返回重复值 frame[dframe.duplicated()]

删除多余的空白字符  tokes = [s.strip() for  s in text.split(',')]  带和

Python strip() 方法用于移除字符串头尾指定的字符
（默认为空格或换行符）或字符序列。

注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。


6,。3.2 映射 映射关系就是创建一个映射关系表；把元素跟一个特定的 标签或字符串绑定起来
一下函数都表示映射关系的dict对象作为参数
replace() 替换
map()
rename()


--------1，用映射替换元素
定义映射关系 就元素作为键星元素作为值
newcolors = {
	'rosso':'red',
	'verd':'green'
}
frame.replace(newcolors)

替换nan元素 ======
ser.replace(np.NaN,0)
ser.replace([11,12],np.nan)
ser.replace([1,2].[np.nan,0])
ser.replace({1:np.nan,0})





------用映射关系添加元素---
frame['price'] = frmae['item'].map(price)  添加price列，以item列作为合并向，
以字典price作为参数

----3.重命轴名索引
reindex = {0:'one',2:'two',3:'three'}
frame.rename(reindex)     #不修改原始数据  
列名要是用columns选项
frame.rename(index= reindex,columns=recolumn)

单个元素替换的简单情况
frame.rename(INDEX={1:'FIRST'},columns={'item':'object'})
----原对象不改变------   设置 inplace=True 就地修改





------6.4 离散化和面元划分
1. 定义一个数组 存储用于面元划分的个数值

bins  = {0,25,50,75,100 }   面元
然后对result数组用 cut()函数，传bins参数
cat = pd.cut(result,bins)
cat.levels   bins空间值
cat.labels  所属面元，可改名字
cat.codes   范围个数
cat.categories  显示范围

参数 precision = 2  小数位数

pd.count_values(cat)  计算数量

还可指定面元的名称 赋给labels属性
pd.cut(results,5) 划分为5部分


---还有 qcut()函数，直接分为5部分，
元素数一样，面元区间大小一样
qu = pd.qcut(results,5)
pd.value_counts(qu)


---异常值检测和过滤---------
frame.std() 每一列的表中差
frame[(np.abs(frame) > (3*frame.std())).any(1)]



-------6.5 随机排序=== 重要点
可以用函数生成一个 随机数组
new_order = np.random.permutation(5)  5ge
frame.take(new_order)
以随机数组作为顺序

new_order = [3,4,2]  部分排序
frame.take(new_order)

随机选行
df.sample(n=3，replace=True) 随机选3行,raplace参数可以重复选择


随机取样-------------
用函数  np.random.randint()
sample = np.random.randint(0,len(frame),size=3)
frame.take(samole)


6.6-----字符串处理----
6.6.1 内置的字符串的处理方式-----
text.split(',')
rokees = [s.strip() for s in text.split(',')]

字符串的拼接---
''.join(strings) ';',','等等

查找子串----
text.index('Boston')  返回索引，找不到会报错
text.find('Boston')  返回索引，找不到返回-1

text.count('e') 计算数量

替换或删除子串---
text.replace('Avenun','Street')
删除 text.replace('1','')


6.6.2 正则表达式------
re.split('\s+',text)

------编译正则表达式 节省内存空间------
regex = re.compile('\s+')
reges.split(text)

---re.findall('A\w+',text) 找到所有的值
search = re.search('[A,a\w]+',text)  返回第一处
search.start() 
search.end() 子串的开始和结束位置

------re.match()  返回一个特殊类型对象
match = re.match('T\w+',text)
text[match.start():match.end()] 返回值

6.7 数据聚合

6.7.1数据分类   GroupBy---------
分组：将数据分成多个组
用函数处理：用函数处理每一组
合并：把不同组的到的结果合并起来

分组：group = frame['price1'].groupby(frame1['color'])
生成组对象
查看分组  group.groups
group.mean()  以颜色为组的平均数

6.7.3 等级分组
ggroup = frame['price1'].groupby(frame1['color'],frame['object'])

一次就分组并计算
fame[['price1','prince2']].groupby(frame['color']).mean()
frame.groupby(fame['color']).mean()

6.8 组迭代
for name,group in frame,groupby('color'):
	print name
	print group

----------6.8.1 链是转换=----------
巨大灵活性
frame.groupby(frame['color'])[price1].mean()

(frame.groupby(frame['color'].mean()).[price1]

给列名添加前缀
means = frae.groupby(color).mean().add_prefix('mean_')

6.8.2 分组函数
quantile() 计算分位数
group['prince1'].quantile(0.6)

还可以自定义函数，传给函数 agg()
group.agg(range) range是自定义函数

还可同时使用多个聚合函数
group['piice1'].agg(['mean','std',range])

df3 = df.groupby(['fruit','color'])['price'].mean()
# type(df1)
# for name,group in df1:
#     print(name)
#     print(group)

df2 = dict(list(df.groupby(by='fruit')))['apple']['price'].sum()
# df2
df.groupby(by='fruit')[['price']].mean() #变成DataFrame对象

语法糖
df['price'].groupby([df['fruit'],df['color']]).mean()
yong自定义函数
def dd(x):
    return x.max() - x.min()
df['price'].groupby(df['fruit']).agg(dd)


df1 = dict(list(df.groupby(by=(['fruit','color']))))
for x in df1:
    print(df1)


6,9 ==========高级数据聚合========重要点
把聚合操作的到的数据放到DataFrame元数据中

sumss = frame.groupby('color').sum().add_prefix('tot_')  可迭代遍历

merge(frame,sums,left_on='color',right_index=True)

transform() 实现数据聚合=======
frrame.groupby('color').transform(np.sum).add_prefix('tot_')
作为参数的函数必须生成一个标量，因为只有这样才能进行广播




====pandas的矢量化函数=====
处理数据时，有些数据类型不一样，跳过不同类型的数据
data.str.contains('gmail') #查找
data.str.splite('@')   #分割

#将col_names转化为list
col_list = col_names.tolist()
col_list

data.ix[:5,:] #取几行几列

data.head(5)  #默认输出前五行
data.tail()   #输出后几行


===pandas的时间序列=====   指在不同时间上收集到的数据
日期和时间数据类型额工具===

from datetime import datetime
now = datetime.now()
now.year
now.month
now.day
7no7.hour


时间详见
datetime(2019,1,10) - datetime(2018,10,10,1,10,10)
d.days
d.seconds

from datetime import timedelta
start = datetime(2017,12,10)
start+timedelta(7)*2   #加到天数上

=====字符串与日期胡转=====
date = datetime(2018,7,5)
sd = str(date) #转字符串
date.strftime('%Y-%m-%d')    #转字符串

v = '2011-05-09'
datetime.strptime(v,'%Y-%m-%d')  #字符串转时间 格式必须
v1 = ['7/4/2012','8/6/2018']
[datetime.strptime(x,'%m/%d/%Y') for x in v1]  #注意格式的不同

from dateutil.parser import parse #zhuan日期 导入包
parse(v)
parse('6/7/2019',dayfirst=True)   #

import pandas as pd 
v = ['2018-4-5 11:23:08','2019-2-4 9:30:00']  
v1 = pd.to_datetime(v + ['NaN']) #类型为DatetimeIndex 注意 加NAN为 空时间
v1.isnull()

ts[::2] #每隔两个区一个

切片选取
stamp=ts.index[2]
ts[stamp]

ts['1/10/2011']
ts['20110110']

指定时间为索引======
import numpy as np
ts = pd.Series(np.random.randn(1000),index = pd.date_range('1/1/2000',periods = 1000))
ts['5/2001']  #quzhi
ts['1/7/2002':]  #qie片
ts['1/7/2002':'8/9/2002']

ts.truncate(after='1/7/2002')  #函数区指定日期之前的数据



=======有重复时间索引======
dates = pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/3/2000'])
ts1 = pd.Series(np.random.randn(4),index=dates)

ts1.index.is_unique  #注意
ts1['1/1/2000']  #取值

ts1['1/2/2000']  #注意是切片
print(ts1.groupby(ts1.index).sum())  #分组
ts1.groupby(level=0).count()  #当分组对象（索引）是重复的时候【即日期可能是重复的】， 那么level = 0则将索引是一样的分开了

生成日期范围=======
pd.date_range('2012-2-3','2012-10-3',freq='BM') 

偏移量
freq='4h' 
#BM表示每个月最后一天  默为D 天  每4小时

移动数据
st = pd.Series(np.random.randn(4),index=pd.date_range('2017-5-1',periods=4))
st.shift(2)  #最上面两个移动到下面，上面连个变为空
st.shift(-2)  #最后连个移动
st.shift(2,freq='M')  #移动了月份



最重要的=====重采样======
频率降低-降采样 反过来-升采样

降采样：
t = pd.DataFrame(np.random.uniform(10,50,(100,1)),
	index=pd.date_range('2017-1-1',periods=100))
t.resample('10D').mean()  ‘M’ 月

升采样：意义不大
s.resample('D').asfreq()  #转换成高频率
s.resample('D').ffill()  #用第一行填充每一行
s.resample('D').ffill(limit=2)   #填充前两行


one_hot 独热编码  是一种映射关系 
str_ =pd.Series(list("abc"))
pd.get_dummies(str_)
	a	b	c
0	1	0	0
1	0	1	0
2	0	0	1

特征相关性
import numpy as np 
data = pd.DataFrame(np.random.rand(10,12))
sns.heatmap(data.corr(),cmap=colormap, annot=True,vmax=1.0,vmin=0.5)
