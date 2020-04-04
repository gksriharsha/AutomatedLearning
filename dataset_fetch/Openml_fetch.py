import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.settings import Settings
import sys,os
from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
#sys.path.append(r'V:\Python\AutomatedLearning\dataset_fetch\learn')

from dataset_fetch.learn.learn.spiders.MetaLinkCollector import dataset_spider
from dataset_fetch.learn.learn.spiders.MetaDataCollector import Dat
import os



def fetch(total_datasets):
    @defer.inlineCallbacks
    def crawl(): 
        os.environ['SCRAPY_PROJECT'] = 'learn'
        settings = get_project_settings()
        
        #settings.setdict
        settings.update({
            'FEED_URI': 'dataset_fetch/meta/Links.csv',
            'FEED_FORMAT': 'csv'
        })
        runner = CrawlerRunner(settings)
        yield runner.crawl(dataset_spider,total_datasets)
        settings = get_project_settings()
        settings.update({
            'FEED_URI': 'dataset_fetch/meta/Data.csv',
            'FEED_FORMAT': 'csv'
            }
        )
        settings.update({'ITEM_PIPELINES':{'dataset_fetch.learn.learn.pipelines.LearnPipeline':200}})
        runner = CrawlerRunner(settings)
        yield runner.crawl(Dat)
        reactor.stop()

    crawl()
    reactor.run()
