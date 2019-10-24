json 本身就是一个字符串
将python对象转换成json用 json.dumps()
dump()函数可以把python对象转换成json字符串。并且可以将其输入到文件中
要关掉ensure_ascii 默认为ascii编码
python中只有int，float，str，list，dict,tuple可以转换为json字符串
with open("person.json",'w',encoding='utf-8') as fp:
	json.dump(object,fp,ensure_ascii=False)


json字符串转换为python对象 
json.loads()#仅仅是转化

#与文件有关
with open('person.json','r',encoding='utf-8') as fp:
	person = json.load(fp)

读csv文件
with open('name.csv','r') as fp:
	#reader是一个迭代器
	reader = csv.reader(fp)
	#从第二行开始读取
	next(reader)
	for x in reader:
		#通过下表获取数据
		name = x[1]
		volumn = x[-1]
		print({'name':name,'volunm':volunm})

	#reader 是一个迭代器，返回一个字典，可以用字典key遍历
	with open('name.csv','r') as fp:
		reader = csv.DictReader(fp) 
		for x in reader:
			value = {'name':x['name']}
			print(value)

写入csv文件，需要创建一个writer对象
headers = ['name','age','country']
与元组的形式
values = [('r',10,'china'),
        ('c',23,'janpan')
         ]
with open("f1.csv",'w',encoding='utf-8',newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(headers)
    writer.writerows(values)

以字典的形式
headers = ['name', 'age', 'country']
    vaiues = [
        {'name':'r','age':'10','city':'china'},
        {'name': 'r', 'age':10,'city': 'china'},
        {'name': 'r', 'age':10,'city': 'china'}
    ]
    #指定打开编码，去除空行
    with open("f1.csv", 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp,headers)
        #写入标题信息
        writer.writeheader()
        writer.writerows(values)

代码链接mysql
mport pymysql
#创建链接对象
conn= pymysql.connect(host='localhost','port'=3306,password='root',db='abname',user='root')
#创建游标对象
cursor = conn.cursor()

#插入数据，输入变化
sql = """
insert into user(id,username,age,password) values(null,%s,%s,%s)
"""
username = 'r'
age = '10'
password = '123'
cursor.execute(sql,(username,age,password))
#增删改后要提交，默认为回滚
conn.commit()
conn.close()

查找 
result = cursor.fetchone() #查找一个
cursor.fetchall() #查找所有
cursor.fetchmany(size) #查找指定适量的数据



