#-*-coding:utf-8-*-

import requests
import lxml.html as H
import string 
import MySQLdb as mdb
#from Beautiful import Beautiful

def get_href(count):
	raw_url='http://roll.news.sina.com.cn/news/gnxw/gdxw1/'
	url=raw_url+'intex_'+str(count)+'.shtml'
	a=requests.get(url)
	raw_content=a.text
	dom=H.document_fromstring(raw_content)
	content_href=dom.xpath("//ul[@class='list_009']/li/a/@href")
#	for href in content_href:
#		print href
	#next_href=raw_url+dom.xpath("//div[@class='pagebox'][1]/span[@class='pagebox_next'][1]/a/@href")[0].split('/')[-1]
	#print next_href
	#return content_href,next_href
	return content_href
#################

def get_content(content_href):
	for href in content_href:
		if href.split('/')[2] != 'news.sina.com.cn':
			continue
		print href
		raw_content=requests.get(href)
		raw_content.encoding='utf-8'
		dom=H.document_fromstring(raw_content.text)

		clas2=clas3=''
		clas1=dom.xpath("//div[@class='bread']/a/text()")
		title=dom.xpath("//div[@class='page-header']/h1/text()")[0]
		content=' '.join(dom.xpath("//div[@id='artibody']/p/text()"))
		keyword=' '.join(dom.xpath("//div[@class='article-keywords']/a/text()"))
		print type(clas1),type(clas2),type(clas3),type(title),type(content),type(keyword)
		conn=mdb.connect(host='localhost',user='root',passwd='',db='news',charset='utf8')
		cur=conn.cursor()
		sql='insert into sina (clas1,clas2,clas3,title,content,keyword) values ("%s","%s","%s","%s","%s","%s")' 

		cur.execute(sql,(clas1,clas2,clas3,title,content,keyword))
		cur.close()
		conn.commit()
		conn.close()

#pages='109'
count = 1
while count < 110:
	print u"正在抓取第 %d 页的内容" % (count)
	content_href=get_href(count)
	get_content(content_href)
	count+=1
