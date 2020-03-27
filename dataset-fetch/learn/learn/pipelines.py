# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


class LearnPipeline(object):
    def open_spider(self, spider):
        try:
            self.file = open('items.txt', 'a')
        except:
            self.file = open('items.txt', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.file.write(str(item))
        self.file.write('\n')
        return item