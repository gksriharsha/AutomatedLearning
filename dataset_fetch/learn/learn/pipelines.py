# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.exceptions import DropItem

class LearnPipeline(object):
    def open_spider(self, spider):
        try:
            self.file = open('items.txt', 'a')
        except:
            self.file = open('items.txt', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        print('##############################################################################################################################')
        if(spider.name == 'CSV'):
            if(len(item['FeatureMissingValues']) < 5):
                f = open('dataset_fetch/meta/dropped.csv','a+')
                f.write(item['MetaData'])
                f.close()
                raise DropItem("Not Enough Features Available")
            else:
                return item