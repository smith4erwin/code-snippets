# -*- coding:utf8 -*-
import scrapy
from yidian.items import YidianItem
from scrapy.http import Request 

cookie={'Cookie': 'JSESSIONID=constant-session-1; Hm_lvt_15fafbae2b9b11d280c79eff3b840e45=1428999876; Hm_lpvt_15fafbae2b9b11d280c79eff3b840e45=1428999876'}
class YidianSpider(scrapy.Spider):
	name='yidian'
	start_urls=['http://www.yidianzixun.com']

	def parse(self,response):
		for id in xiaolei_id.a:
			xiaolei_url='http://www.yidianzixun.com/home?page=channel&'+'id'
			yield Request(xiaolei_url,callback=self.parse_xiaolei_name,cookies=cookie)

	def parse_xiaolei_name(self,response):
		





