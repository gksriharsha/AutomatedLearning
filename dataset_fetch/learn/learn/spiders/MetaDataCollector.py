import scrapy
from ..items import LearnItem
import pandas as pd
import uuid,csv

class Dat(scrapy.Spider):
	name = 'CSV'
	
	start_urls = [
		'https://www.openml.org/d/31'
	]
	

	def __init__(self):
		super().__init__()
		
		self.datasets = pd.read_csv('dataset_fetch/meta/Links.csv')

	def detailed_data(self,response):

		id = uuid.uuid4()

		NumberOfProperties = int(response.xpath("//div[@class='tab-pane active']/h3[2]/text()").extract_first() \
		                         .strip('properties').strip())
		
		Metadata = {}
		Metadata.__setitem__('fID',id)
		for i in range(NumberOfProperties):
			key = response.xpath(f'//*[@id="data_overview"]/div[7]/div[{i+1}]/div[1]/a/text()').extract()[1].strip()
			print(key)
			value = response.xpath(
				f"//div[@class='tab-pane active']/div[7]/div[{i+1}]/div[2]/text()").extract_first().strip()
			print(value)
			# //*[@id="data_overview"]/div[7]/div[1]/div[1]/a/text()
			# /html/body/div[4]/div[4]/div/div/div[1]/div[7]/div[1]/div[1]/a/text()
			
			# //*[@id="data_overview"]/div[7]/div[6]/div[1]/a/text()
			Metadata.__setitem__(key, value)

		try:
			with open('dataset_fetch/meta/MetaData.csv','r') as f:
				pass
			with open('dataset_fetch/meta/MetaData.csv','a') as f:
				reader = csv.DictWriter(f,fieldnames=Metadata.keys())
				reader.writerow(Metadata)
		except:
			with open('dataset_fetch/meta/MetaData.csv','w') as f:
				writer = csv.writer(f)
				writer.writerow(Metadata.keys())
				writer.writerow(Metadata.values())
		return id
	
	def parse(self, response):
		
		#print(response)
		
		datasetinfo = LearnItem()
		
		link = response.xpath("//div[@class='tab-pane active']/ul[@class='hotlinks']/li[2]/a/@href").extract_first()
		
		filename = response.xpath("//div[@class='tab-pane active']/h1[@class='pull-left']/text()").extract_first().strip()
		
		label_header = response.xpath("//div[@class='tab-pane active']/div[@class='cardtable']" +
		                              "/div[@class='features hideFeatures']" +
		                              "/div[@class='table-responsive']" +
		                              "/table[@class='table']/tr[1]/td[1]/text()").extract_first().strip('()')
		
		datasetinfo['Metadata'] = self.detailed_data(response)
		
		datasetinfo['csv_url'] = link
		
		datasetinfo['dataset_url'] = response.url
		
		datasetinfo['Filename'] = filename
		
		datasetinfo['Label_header'] = label_header
		
		number_of_features = int(response.xpath(r'//*[@id="data_overview"]/h3[1]/text()').extract_first().split(' features')[0])
		
		datasetinfo['FeatureName'] = []
		datasetinfo['FeatureType'] = []
		datasetinfo['FeatureUniques'] = []
		datasetinfo['FeatureMissingValues'] = []
		for i in range(number_of_features):
			featureName = response.xpath(
				f'//*[ @ id = "data_overview"]/div[4]/div/div/table/tr[{i+1}]/ td[1]/text()').extract_first().strip()
			featureType = response.xpath(f'//*[@id="data_overview"]/div[4]/div/div/table/tr[{i+1}]/td[2]/text()').extract_first().strip()
			featureUniques = int(response.xpath(f'//*[@id="data_overview"]/div[4]/div/div/table/tr[{i+1}]/td[3]/text()[1]').extract_first().strip('unique values'))
			featureMissingValues = int(response.xpath(f'//*[@id="data_overview"]/div[4]/div/div/table/tr[{i+1}]/td[3]/text()[2]').extract_first().strip('missing'))
			
			
			datasetinfo['FeatureName'].append(featureName)
			datasetinfo['FeatureType'].append(featureType)
			datasetinfo['FeatureUniques'].append(featureUniques)
			datasetinfo['FeatureMissingValues'].append(featureMissingValues)
		
		yield datasetinfo
		
		for URL in self.datasets.loc[:, 'dataset_url']:
			yield response.follow(URL, callback=self.parse)

	