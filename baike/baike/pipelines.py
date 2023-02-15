# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
from baike.items import BaikeItem
import MySQLdb as mdb
class BaikePipeline(object):
    def __init__(self):
        self.conn=mdb.connect(host='localhost',user='root',passwd='',db='news',charset='utf8')
    def process_item(self, item, spider):
#        print type(item['url']),type(item['title']),type(item['poly']),type(item['tag'])
#        with open('1.txt','a+') as f:
#            f.write('\n')
#            f.write(item['url'].encode('gbk'))
#            f.write('\n')
#            f.write(item['title'].encode('gbk'))
#            print type(item['title'])
#            f.write('\n')
#            f.write(item['poly'].encode('gbk'))
#            f.write('\n')
#            f.write(item['tag'].encode('gbk'))
#            f.write('\n\n\n')


  
#        conn=mdb.connect(host='localhost',user='root',passwd='',db='news',charset='utf8')
        cur=self.conn.cursor()
        sql='insert into baike (url,title,poly,tag) values ("%s","%s","%s","%s")' 
        cur.execute(sql,(item['url'].encode('utf-8'),item['title'].encode('utf-8'),item['poly'].encode('utf-8'),item['tag'].encode('utf-8')))
        cur.close()
        self.conn.commit()
        return item
