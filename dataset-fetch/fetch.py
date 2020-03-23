import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import sys,os
from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
#sys.path.append(r'V:\Python\AutomatedLearning\dataset-fetch\learn')

from learn.learn.spiders.MetaLinkCollector import dataset_spider
from learn.learn.spiders.MetaDataCollector import Dat

# process = CrawlerProcess(settings={
#     'FEED_FORMAT':'csv',
#     'FEED_URI':'../meta/Links.csv'
# })
# process.crawl(dataset_spider)
# process.start(stop_after_crawl=False) # the script will block here until all crawling jobs are finished
# process = CrawlerProcess(settings={
#     'FEED_FORMAT':'csv',
#     'FEED_URI':'../meta/Data.csv'
# })
# process.crawl(Dat)
# process.start()




@defer.inlineCallbacks
def crawl(): 
    runner = CrawlerRunner({
        'FEED_URI': '../meta/Links.csv',
        'FEED_FORMAT': 'csv'
    })
    yield runner.crawl(dataset_spider)
    runner = CrawlerRunner({
        'FEED_URI': '../meta/Data.csv',
        'FEED_FORMAT': 'csv'
    })
    yield runner.crawl(Dat)
    reactor.stop()

crawl()
reactor.run()
