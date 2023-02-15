# -*- coding:utf8 -*-
import scrapy
from baike.items import BaikeItem
from scrapy.http import Request 

class BaikeSpider(scrapy.Spider):
	name='baike'
	start_urls=['http://baike.baidu.com/view/3140664.htm']

	def parse(self,response):
#		items=[]
		
		raw_fenlei_urls=list(set(response.xpath('//a[contains(@href,"/fenlei/")]/@href').extract()))
		raw_view_urls=list(set(response.xpath('//a[contains(@href,"/view/")]/@href').extract()))
		raw_subview_urls=list(set(response.xpath('//a[contains(@href,"/subview/")]/@href').extract()))

		for raw_fenlei_url in raw_fenlei_urls:
			lst=raw_fenlei_url.split('/')
			fenlei_url="http://baike.baidu.com/fenlei/"+lst[-1]
			yield Request(fenlei_url)

		for raw_view_url in raw_view_urls:
			lst=raw_view_url.split('/')
			view_url="http://baike.baidu.com/view/"+lst[-1]
			yield Request(view_url,callback=self.parse_view)

		for raw_subview_url in raw_subview_urls:
			lst=raw_subview_url.split('/')
			subview_url="http://baike.baidu.com/subview/"+lst[-2]+'/'+lst[-1]
			yield Request(subview_url,callback=self.parse_subview)


	def parse_view(self,response):
		
		item=BaikeItem()
		item['url']=response.url
		item['title']=' '.join(response.xpath('//span[@class="lemmaTitleH1"]/text()').extract())
		item['tag']=' '.join(response.xpath('//sapn[@class="taglist"]/text()').extract())
		item['poly']=''
		yield item

		raw_fenlei_urls=list(set(response.xpath('//a[contains(@href,"/fenlei/")]/@href').extract()))
		raw_view_urls=list(set(response.xpath('//a[contains(@href,"/view/")]/@href').extract()))
		raw_subview_urls=list(set(response.xpath('//a[contains(@href,"/subview/")]/@href').extract()))

		for raw_fenlei_url in raw_fenlei_urls:
			lst=raw_fenlei_url.split('/')
			fenlei_url="http://baike.baidu.com/fenlei/"+lst[-1]
			yield Request(fenlei_url)

		
		for raw_view_url in raw_view_urls:
			lst=raw_view_url.split('/')
			view_url="http://baike.baidu.com/view/"+lst[-1]
			yield Request(view_url,callback=self.parse_view)

		
		for raw_subview_url in raw_subview_urls:
			lst=raw_subview_url.split('/')
			subview_url="http://baike.baidu.com/subview/"+lst[-2]+'/'+lst[-1]
			yield Request(subview_url,callback=self.parse_subview)


	def parse_subview(self,response):

		item=BaikeItem()
		item['url']=response.url
		item['title']=' '.join(response.xpath('//span[@class="lemmaTitleH1"]/text()').extract())
		item['tag']=' '.join(response.xpath('//sapn[@class="taglist"]/text()').extract())
		item['poly']=' '.join(response.xpath('//span[@class="polysemeTitle"]/text()').extract())
		yield item


		raw_fenlei_urls=list(set(response.xpath('//a[contains(@href,"/fenlei/")]/@href').extract()))
		raw_view_urls=list(set(response.xpath('//a[contains(@href,"/view/")]/@href').extract()))
		raw_subview_urls=list(set(response.xpath('//a[contains(@href,"/subview/")]/@href').extract()))

		for raw_fenlei_url in raw_fenlei_urls:
			lst=raw_fenlei_url.split('/')
			fenlei_url="http://baike.baidu.com/fenlei/"+lst[-1]
			yield Request(fenlei_url)

		
		for raw_view_url in raw_view_urls:
			lst=raw_view_url.split('/')
			view_url="http://baike.baidu.com/view/"+lst[-1]
			yield Request(view_url,callback=self.parse_view)

		
		for raw_subview_url in raw_subview_urls:
			lst=raw_subview_url.split('/')
			subview_url="http://baike.baidu.com/subview/"+lst[-2]+'/'+lst[-1]
			yield Request(subview_url,callback=self.parse_subview)

