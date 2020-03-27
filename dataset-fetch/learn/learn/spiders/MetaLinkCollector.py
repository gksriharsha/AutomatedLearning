import scrapy
import time
from ..items import LearnItem
import uuid
import sys,os

class dataset_spider(scrapy.Spider):
	name = 'Datasets'
	
	start_urls = [
		'https://www.openml.org/search?type=data&from=0'
	]
	
	Datasets = 100
	
	def parse(self, response):
		
		datasetinfo = LearnItem()
		
		for dataset in response.xpath("//div[@class='searchresult panel']"):
			Filename = dataset.xpath(".//div[@class='itemhead']/a/text()").extract_first()
			URL = dataset.xpath(".//div[@class='itemhead']/a/@href").extract_first()
			datasetinfo['Filename'] = Filename
			datasetinfo['dataset_url'] = 'https://www.openml.org/'+URL
			yield datasetinfo
		
		next_page = 'https://www.openml.org/search?type=data'+f'&from={dataset_spider.Datasets}'
		
		if dataset_spider.Datasets < 2400:
			dataset_spider.Datasets = dataset_spider.Datasets + 100
			yield response.follow(next_page,callback = self.parse)

if(__name__ == '__main__'):
    os.chdir(r'Q:\Python Workspace\AutomatedLearning\dataset-fetch\learn\learn')
    os.system('scrapy crawl Datasets -o ../meta/MetaLink.csv')