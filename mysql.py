1.查看服务器编码  show variables like 'char%';
2.查看创建表的信息  show create table 表名；
3.更改表的编码  alter table 表名 character set utf8;
4. 改列的编码  alter table 表名 change 列名 列名 
varchar(20) character set utf8 not null；

解决多张拥有数据的编码 
1.导出表结构 导出数据 mysqldump
2.删除原有的数据库 重新创建 指定编码
3.改配置文件 导入数据表

session会话变量 局部变量
show session variables;
更改 set variables名 = "on"

全局变量  show global variables;
 set global variables名 = "off"

函数 now()

create table student1 (
	id INT NOT NULL  AUTO_INCREMENT PRIMARY KEY,
  s_name VARCHAR(20)  NOT NULL,
  nickname VARCHAR(20) NULL,
  sex CHAR(1) NULL,
  in_time DATETIME NULL
)DEFAULT CHARSET 'utf8';

show create table student;

insert into student1(s_name,nickname,sex,in_time) VALUE('张三','三个','男',NOW());

select  * from student;

SELECT * from student1 where sex='男' ORDER BY id desc LIMIT 0,5;
update student1 set sex='女' where sex='男';
delete from student1 where sex='女';

