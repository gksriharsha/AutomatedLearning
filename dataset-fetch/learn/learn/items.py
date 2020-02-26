# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field


class LearnItem(Item):
    # define the fields for your item here like:
    Filename = Field()
    dataset_url = Field()
    csv_url = Field()
    Label_header = Field()
    Metadata = Field()
    FeatureName = Field()
    FeatureType= Field()
    FeatureUniques= Field()
    FeatureMissingValues= Field()
    
