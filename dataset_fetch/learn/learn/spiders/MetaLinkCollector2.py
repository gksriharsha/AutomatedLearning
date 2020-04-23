import scrapy
import time
from ..items import SourceLinkItem
import uuid
import sys,os
import time

class dataset_spider(scrapy.Spider):
	name = 'Datasets_100'
	
	start_urls = [
		'https://www.openml.org/s/225/data'
	]

	def parse(self, response):

		datasetinfo = SourceLinkItem()

		names = response.xpath('//*[(@id = "itempage")]//a/text()').getall()
		urls = response.xpath('//*[(@id = "itempage")]//a/@href').getall()
		pairs = list(zip(names,urls))
		print(pairs)
		for (name,url) in pairs:
			datasetinfo['Filename'] = name
			datasetinfo['dataset_url'] = 'https://www.openml.org/'+url
			yield datasetinfo