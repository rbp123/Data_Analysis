pandas 数据读写
5.1 I/O APL工具
--------数据分析读写工具 I/O APL函数
读取函数
===read_csv _execl _hdf _sql _json _html 
_stata _clipboard _pickle
写入函数
===to_csv  _execl _hdf _sql _json _html 
_stata _clipboard _pickle

---5.2 CSV和文本文件
csv = read_csv('myCSV_01')  返回 DataFrame对象
既然csv文件被视作文本文件，还可以用 read_table()函数，但是得指定分隔符
pd.read_table('ch05_01.csv',sep=',') 返回 DataFrame对象

以上是有表头的，----对于没有表头的，使用headers选项，pandas 会添加默认表头
read_csv('myCSV_01',headers=None)
还可以用names选项指定表头，为表头赋值
read_csv('myCSV_01',names=['whitle','red','bule','green','animal'])
生成等级结构
read_csv('myCSV_01',index_col=['color','status'])

list(open('ex3,csv'))  #有换行符

有转义字符，前面加r

5.3.1  用RdgExp解析TXT文件
数据之间以不同的符号分隔开，用正则

例子 所有数据都=====以默认=====制表符====和换行符隔开
read_table('myCSV_01',sep='\D*',headers=None)  #\D*正则匹配多个非数字字符
返回 DataFrame对象

排除多余的行
排除前五行 skiprows = 5
排除第五行 skiprows = [5]
read_table('myCSV_01',sep=',',skiprows=[1,2,9]) 排除三行

---5.3.2 从TXT文件中读取部分数据
用skiprows 起始行 和nrows 从起始行开始读几行 选项
read_csv('myCSV_01',skiprows[2],nrows=3) 

--- 5.3.3 往csv文件写数据
frame.t0_csv('ch05_07.csv')   默认写入index和列名
用参数可改变默认行为
frame.t0_csv('ch05_07.csv'，index=Flase,header=Flase)
-------数据中的NaN写入后是空字段------
可以使用na_rep参数把空字段替换掉 0，NULL，NaN
frame.to_csv('ch05_07.csv',na_rep='0') 

---所有的函数和选项也适用于Series------

5.4 读写HTML文件  安装html5lib
5.4。1  写入数据到HTML文件
------to_html()函数可以直接把 DataFrame对象转换成HTML表格
frame = pd.DataFrame(np.arange(4).reshap(2,2))
print(frame.to_html())

--------演示在thml 文件中自动生成表格
frame = pd.DataFrame(np.random.random((4,4)),index=['whitle','black',
	'red','blue'],columns=['up','down','right','left'])
---省略
s.append(frame.to_html())
-----省略
HTML = ''.json(s)

html_file = open('myFrame.html','w')
html_file.write(html)
html_file.close()
浏览器打开 myFrame.html DataFrame数据会以表格的形式显示在浏览器左上方

5.4.2 从HTML文件中读取数据
read_html()函数 解析HTML文件，寻找HTML表格，找到转换成 DataFrame对象返回

web_frames = pd.read_html('myFrame.html')
web_frames[0]  第一个HTML表格

-----直接从网页解析表格------
ranking = pd.read_html('url地址') 
ranking[0]  取第一个

-----5.5 从XML读取数据 
pandas没有专门处理XML的函数，但python有很多读写XML格式数据的库

----lxml库   在大文件处理方面性能优越
直接把XML的数据结构转换成 DataFrame对象，要用到楼下买了库的二级模块 objectify
from lxml import objectify
xml = objectify.parse('book.xml')  先用parse解析XML文件

root = xml.getroot()   获取树结构的各个节点

获取标签，用点号隔开
root.Book.Author
'Ross,Mark'
root.Book.PublshDate
'2014-22-01'

同时获取多个元素，函数 getchildren() 能同时获取某个元素的所有子节点
root.getchildren()
再用tag属性，获取到子节点的tag属性的名称
[child.tag for child in root.Book.getchildren()]
['Author','Title','Genre','Price','PublshDate']
用text属性，可获取到位于标签之间的内容
[child.text for child in root.Book.getchildren()]
['Ross,Mark','XML Cookbook','Computer','23.56','2014-22-01']

--把树结构转换成 DataFrame对象

------5.6 读写Excel文件-------
read_execl()函数 将execl文件转换成 DataFrame对象
pd.read_execl('data.xls')  第一个工作表
pd.read_execl('data.xls','sheet2'或2)

写入函数
frame.to_excel('data2.xls')


------5.7 JSON数据
写入函数
frame.to_json('frame.json')
读取函数
pd.read_json('frame.json')

=======res = json.loads(odj) #将json字符串转换成python形式
res1 = json.dumps(res)  #序列化，转换成DataFrame对象

-----pandas 的 json_normalize() 函数能将字典或列表转换成表格
from pandas.io.json import json_normalize

更复杂的结构，就无法使用 read_json()函数来处理

file = open('books.json','r')
text  = file.read()
text = json.loads(text)
转换成了一个字符串

json_normalize(text,'books')  查看图书信息表格，读取以books作为键的元素的值

----5.8 HDF5格式 （二进制） =====分析大量数据======
HDF5库 
from pandas.io.pytables import HDFStore
创建一个.h5文件，把 DataFrame对象存储
store = HDFStore('mydata.h5')
store['obj1'] = frame
还可以吧多种数据结构的数据存储到一个 HDF5文件中

两种存储模式 ；fixed和table 后者有点慢
store.put('obj2',frame,format='table')

store.select('obj2',where=['index>=10 and index <= 15'])

逆操作
store['obj2']


5.9 pickle  python对象序列化======
序列化是指把对象的层级几个转换成字节流的过程
 
 5.9.1 用 cPickle实现python对象序列化
 import  cpickle as pickle
 dumps()函数
 pickled_data = pickle.dumps(data)

 数据序列化后，再写入文件或用套接字，管道等发送就很简单

 反序列化 loads()函数
 nframe = pickle.loads(pickled_data)


 ----5.9.2 用pandas实现对象序列化-----  ‘’‘重点====
 无需导入cpickle模块，所有的操作都是隐士进行的
 frame.to_pickle('frame.pkl')
 读取数据
 pd.read_pickle('trame.pkl')





  --------5.10 对接数据库------
  from sqlalchemy import create_engine

  PostgreSQL:
  engine = create_engine('postgresql://scott:tiger@localhost:5432/mydatabase')
  MYSQL:
  engine = create_engine('mysql+mysqldb://scott:tiger@localhost/foo')
  engine = create_engine('mysql+pymysql://root:root@localhost:3306/book')========
  Oracle:
  engine = create_engine('Oracle://scott:tiger@127.0.0.1:1521/sidname')
  MSSQL:
  engine = create_engine('mssql+pyodbc://mydsn')
  SQLite:
  engine = create_engine('sqlite:///foo.db')



  -----5.10.1 SQLite3数据读写 python内置数据库
  链接数据库
engine = create_engine('sqlite:///foo.db')
把 DataFrame对象转换成数据库表
frame.to_sql('colors',engine) 参数为表名和 engine实例
读取数据库
pd.read_sql('colors',engine)  参数为表名和 engine实例

--------pandas是读写数据库的好工具
不使用I/O API的情况
很麻烦

----5.11 NoSQL数据库 MongoDB数据读写------
import pymongo 
client = MongoClient('localhost',27017)
定义数据库
db = client.mydatabase mydatabase数据库名
定义集合（表）
collection = db.mycollection

----------添加到集合之前，DataFrame读写必须转换成JSON格式
import json
record = json.loads(frame.T.to_json()).values()  字符串
collection.mydocument.insert(record)

逆过程 
cursor = collection['mydocument'].find()
dataframe = (list(cursor))
del dataframe['_id'] 删除用作MongoDB内部索引的ID编号这一列

from sqlalchemy import create_engine
import pymysql
import pandas as pd 
engine = create_engine('mysql+pymysql://root:root@localhost:3306/book')
username:password@localhost:3306/dbname'





-----深入pandas------
6.1 数据准备
----1.合并   pandas.merge(frame1,frame2,on='id') 根据一个或多个键链接多行
on选项指明要合并依据的准则列
如果两个Dataframe的列名不同，用left_on和right_on指定基准列
pandas.merge(frame1,frame2,left_on='id',right_on='sid')
连接类型由how选项指定  左连接 有链接 外连接
pandas.merge(frame1,frame2,on='id'，how='outer'或'left'或'right')
要合并多个键
pandas.merge(frame1,frame2,on=['id','sid'],how='outer')
根据索引合并
pd.merge(frame1,frame2,right_index=True,left_index=True)
----但是DataFrame对象的 join()跟适合个人呢就索引进行合并，我们
可以用他合并多个索引相同或索引相同但列却不一致的对象
frame1.join(frame2)


------2.拼接   pandas.concat() 按照轴把多个对象拼接起来
pd.concat([ser1,ser2]) 默认按照axis=0 这条轴拼接数据，如果指定
axis=1,返回的是 DataFrame对象 没重合数据，其实执行的是外链接
设置 join 选项，内连接
pa.concat([ser1,ser2],axis=1,join='inner')
keys选项 设置识别被拼接的部分
pa.concat([ser1,ser2],keys=[1,2])

DataFrame对象的拼接
pd.concat([frame1,frame2],axis=1) 返回 DataFrame对象


------3.结合   pandas的DataFrame.combine_first() 从另一个数据结构获取数据
链接重合的数据，以填充缺失值
还有一种情况，我们无法通过合并或拼接方法组合数据，例如，两个数据集的索引完全或部分重合
combine_first()还可以合并ser对象
ser1.combine_first(ser2)
部分合并
ser1[:3].combine_first(ser2[:3])


6.2.2 轴向旋转
1.按等级索引旋转
入栈  旋转数据结构，把列转换为行
出栈  把行转换成列

队Dataframe对象使用 stack()函数会吧列转换成行，变成Series对象
frame1.stack()
还原为 DataFrame对象
ser.unstack()   还可以传入表示层级的编号和名称
ser.unstack(0)

2.从‘长’格式向‘宽’格式旋转
pivot()函数以用作键的一列或多列作为参数
wideframe = longframe.pivot('color','item')
主键color，第二主键item

6.2.3 删除
删除一列 
del frame1['ball']
删除行 
frame.drop('whitle')

------6.3数据转换
6.3.1 删除重复元素
duplicated() 检测重复的行，返回元素为布尔型的series对象
frame.duplicated()
寻找重复的行  
frame[frame.duplicated()]
删除重复的行  drop_duplicates() 返回删除重复行的 DataFrame对象
tokens = [s.strip() for s in text.split(',')]

---6.3.2 映射
要定义映射关系，最好的对象莫过于dict
1.用映射替换元素
要用新元素代替不正确的元素，需要定义一组映射关系，就元素作为键，新元素作为值
newcolors为一组映射关系
frame.replace(newcolors)

把NaN替换成其他元素
ser.replace(np.nan,0)

-----2.用映射添加元素
price 为dict对象 map()函数可用于 Ser或D对象的一列，以一个函数或字典为参数
在 DataFrame的item这一列应用映射关系，用price作为参数，为 DataFrame对象添加price列
frame[price]  = frame['item'].map(price)
 
----3.重命名轴索引
pandas的 rename()函数，以表示映射关系的字典对象作为参数，替换轴的索引标签
reindex 是以旧索引为键，新索引为值得dict对象
frame.rename(reindex)
若要重命名格列，必须使用column是选项
recolumn 是以旧索引为键，新索引为值得dict对象
frame.rename(inde=reindex，columns=recolumn)

单个索引替换
frame.rename(index={1:'first'},columns={'item':'object'})

以上替换的是映射，而不是原始数据，若要改变。可使用inplace选项，并将其值zhi为True


----6.4 离散化和面元划分
用pandas划分面元之前，首先要定义一个数组，存储用于面元换分的各参数
bins = [0,25,50,75,100]
然后对result3数组应用 cut()函数，同时传入bins变量作为参数，每个元面的个体数不相同
cat = pd.cut(results,bins) 返回的对象为Categorical（类别型）类型

quintiles = pd.qcut(results,5)  这个函数把样本分成5个元 面，保证每个元面的个体数相同，但每个元面的区间大小不同
pd.value_counts(quintiles) 计算所属区间的数据个数

-----异常值得检测和过滤------
randframe = pd.DataFrame(np.random.randn(1000,3))
randframe.describle()
randframe.std()   表中差
用 any()函数，对每一列应用筛选条件
randframe[(np.abs(randframe) > (3*randframe.std())).any(1)]

----6.5 排序
创建一个元素为整数切按升序排列的 DataFrame对象
nframe = pd.DataFrame(np.arange(25).reshape(5,5))
用 permutation()函数创建一个包含0-4这五个数的随机数组
new_order = np.random.permutation(5)
对 DataFrame对象的所有行应用 take()函数，把新的次序传给他
nframe.take(new_order)

还可以对特定的部分行排序 
new_order = [3,4,2]
nframe.take(new_order)

---随机取样-----
sample =np.random.randint(0,len(nframe),size=3)
nframe.take(sample)


--6.6 字符串的处理
6.6.1 内置的字符串处理方法
text.split()函数实现切分
text.strip()删去空白字符（包括换行符）
tokens = [s.strip() for s in text.split(',')]

----拼接多个字符
';'.join(strings) 以；符号隔开
’','.join(strings) 以,符号隔开

---查找字符串
'string' in text
>>>True
text.index('string') 找不到会报错
text.find('string')  找不到会符号-1
这两个函数都符号字符串的索引
--计算出现的次数
text.count('string')

-----替换或删除
text.replace('old_string','new_string')
用空字符串替换，相当于删除
text.replace('1','')

