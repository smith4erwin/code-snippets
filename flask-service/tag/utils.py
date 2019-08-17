# -*- coding: utf-8 -*-

import os

from config import MYPATH


with open(os.path.join(MYPATH.RESOURCE_PATH, 'tags/unchosen_tags.txt')) as f:
    lines = f.readlines()
    unchosen_tags = [line.rstrip('\n') for line in lines]
    unchosen_tags = set(unchosen_tags)

#################################
def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False

def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


#################################
def is_en(word):
    if len(word) == 0:
        return False
    for uchar in word:
        if not is_alphabet(uchar):
            return False
    return True

def is_en1(word):
    if len(word) == 0:
        return False
    # 英文
    # 英文+数字
    b = False
    for uchar in word:
        if is_alphabet(uchar):
            b = True
        if not is_alphabet(uchar) and not is_number(uchar):
            return False
    return b if b else False

def is_zh(word):
    if len(word) == 0:
        return False
    for uchar in word:
        if not is_chinese(uchar):
            return False
    return True

def is_zh1(word):
    if len(word) == 0:
        return False
    # 中文
    # 中文+英文
    # 中文+数字
    # 中文+英文+数字
    b = False
    for uchar in word:
        if is_chinese(uchar):
            b = True
        if not is_chinese(uchar) and not is_alphabet(uchar) and not is_number(uchar):
            return False
    return b if b else False



#################################
def proc1(tag):
    return [tag] if tag != 'tag' else ['']

def proc2(tag):
    filt_words = ['壁纸', '高清', '锁屏', '滚屏', '手机', '安卓', '图片']
    b = [x for x in filt_words if x in tag]
    while len(b) != 0:
        for bb in b:
            tag = ''.join(tag.split(bb))
        b = [x for x in filt_words if x in tag]
    return [tag]

def proc3(tag):
    if '日历' in tag:
        return tag.split('日历') + ['日历']
    return [tag]

def proc4(tag):
    if '明星' in tag:
        return tag.split('明星') + ['明星']
    return [tag]

def proc5(tag):
    en_prunc = list('~`!@#$%^&*()_+-={}|:"?><,./;\'\[]')
    b = [x for x in en_prunc if x in tag]
    while len(b) != 0:
        for bb in b:
            tag = ''.join(tag.split(bb))
        b = [x for x in en_prunc if x in tag]
    return [tag]

def proc6(tag):
    zh_prunc = list('~·！@#￥%……&*（）——+=-【】、；‘’，。、？》《：“”{}|')
    b = [x for x in zh_prunc if x in tag]
    while len(b) != 0:
        for bb in b:
            tag = ''.join(tag.split(bb))
        b = [x for x in zh_prunc if x in tag]
    return [tag]


def filtr(tag):
    filter_word = set(['tag', 'License', 'Commons', 'Creative', 'CC0', 'of', 'Of', 'to', 'the', 'The', 'a', 'A', '', \
               '大陆', '欧美', '港台', '香港', '台湾', '韩国', '日本', '美国', '中国香港', \
               '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月', \
               '一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月', \
               '官方', '其它', '其他', '其他类别', '版权'])
    if tag in filter_word:
        return False
    return True


def process(proc_func, tag):
    a = []
    for tg in tag:
        a.extend(proc_func(tg))
    return a


def unique(tags):
    tags_set = set(tags)
    a = []
    for tag in tags:
        if tag in tags_set:
            a.append(tag)
            tags_set.remove(tag)
    return a


def mapping(tags):
    tags_proc = []
    for tag in tags:
        tag = [tag]
        for proc in [proc2, proc3, proc4, proc5, proc6]:
            tag = process(proc, tag)
        tags_proc.extend(tag)
    tags_proc = unique(tags_proc)
    tags_proc = list(filter(filtr, tags_proc))
    if len(tags_proc) == 0:
        tags_proc = ['']
    return tags_proc
