# -*- coding: utf-8 -*-

# Scrapy settings for baike project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'baike'

SPIDER_MODULES = ['baike.spiders']
NEWSPIDER_MODULE = 'baike.spiders'
ITEM_PIPELINES={'baike.pipelines.BaikePipeline':300}

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'baike (+http://www.yourdomain.com)'
