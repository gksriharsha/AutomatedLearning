import scrapy
import time
from ..items import SourceLinkItem
import uuid
import sys,os
import time

class dataset_spider(scrapy.Spider):
	name = 'Datasets'
	
	start_urls = [
		'https://www.openml.org/search?type=data&from=0',
		# 'https://www.openml.org/search?type=data&from=100',
		# 'https://www.openml.org/search?type=data&from=200',
		# 'https://www.openml.org/search?type=data&from=300',
		# 'https://www.openml.org/search?type=data&from=400',
		# 'https://www.openml.org/search?type=data&from=500',
		# 'https://www.openml.org/search?type=data&from=600',
		# 'https://www.openml.org/search?type=data&from=700',
		# 'https://www.openml.org/search?type=data&from=800',
		# 'https://www.openml.org/search?type=data&from=900',
		# 'https://www.openml.org/search?type=data&from=1000',
		# 'https://www.openml.org/search?type=data&from=1100',
		# 'https://www.openml.org/search?type=data&from=1200',
		# 'https://www.openml.org/search?type=data&from=1300',
		# 'https://www.openml.org/search?type=data&from=1400',
		# 'https://www.openml.org/search?type=data&from=1500',
		# 'https://www.openml.org/search?type=data&from=2100',
		# 'https://www.openml.org/search?type=data&from=1600',
		# 'https://www.openml.org/search?type=data&from=1700',
		# 'https://www.openml.org/search?type=data&from=1800',
		# 'https://www.openml.org/search?type=data&from=1900',
		# 'https://www.openml.org/search?type=data&from=2000',
		# 'https://www.openml.org/search?type=data&from=2200',
	]
	
	def __init__(self,datasets):
		self.total_Datasets = 100
		self.Datasets = 100
	
	def parse(self, response):
		
		datasetinfo = SourceLinkItem()
		
		for dataset in response.xpath("//div[@class='searchresult panel']"):
			Filename = dataset.xpath(".//div[@class='itemhead']/a/text()").extract_first()
			URL = dataset.xpath(".//div[@class='itemhead']/a/@href").extract_first()
			datasetinfo['Filename'] = Filename
			datasetinfo['dataset_url'] = 'https://www.openml.org/'+URL
			yield datasetinfo
		
		# next_page = 'https://www.openml.org/search?type=data'+f'&from={dataset_spider.Datasets}'
		
		# if dataset_spider.Datasets < self.total_Datasets:
		# 	time.sleep(10)
		# 	dataset_spider.Datasets = dataset_spider.Datasets + 100
		# 	yield response.follow(next_page,callback = self.parse)

if(__name__ == '__main__'):
    os.chdir(r'Q:\Python Workspace\AutomatedLearning\dataset_fetch\learn\learn')
    os.system('scrapy crawl Datasets -o ../meta/MetaLink.csv')