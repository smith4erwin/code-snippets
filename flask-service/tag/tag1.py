# -*- coding: utf-8 -*-

import os

import yaml
import gensim
import numpy as np

from config import *
import indexer

from .utils import *

class Tag1(object):
    def __init__(self, idxer=None):
        self.idxer = idxer


    def readcfg(self, cfg):
        with open(cfg, 'r') as f:
            self.args = yaml.load(f)
        logger.info('=== tagger config ===')
        logger.info('\n'+yaml.dump(self.args))

        # general property
        assert 'tag_algo'  in self.args
        assert 'tagdir'    in self.args
        assert 'params'    in self.args
        assert 'n_feature' in self.args

        # own property
        assert 'idx_algo' in self.args['params']
        assert 'idxdir' in self.args['params']

        if self.idxer is None:
            self.idxer = getattr(indexer, self.args['params']['idx_algo'])()
            self.idxer.readcfg(os.path.join(self.args['params']['idxdir'], 'idx.yaml'))
            self.idxer.load()
            
        assert self.idxer.args['idx_algo'] == self.args['params']['idx_algo']
        assert self.idxer.args['idxdir']   == self.args['params']['idxdir']

        imglst_file = os.path.join(self.args['tagdir'], 'img.lst')
        if not os.path.exists(imglst_file):
            open(imglst_file, 'w').close()
        with open(imglst_file, 'r') as f:
            lines = f.readlines()
            self.imglst = [line.rstrip('\n').split('\t') for line in lines]

        self.data = {}
        for info_id, pic_url, tags in self.imglst:
            self.data[info_id] = {}
#           self.data[info_id]['pic_url'] = pic_url
            self.data[info_id]['tags'] = tags.split(' ')
        
        logger.info('{0} images have been tagged'.format(len(self.data)))

        ext_imglst_file = os.path.join(self.idxer.extor.args['vecdir'], 'img.lst')
        with open(ext_imglst_file, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n').split('\t') for line in lines]
#       self.ext_infoid_dict = {}
#       for i, [info_id, pic_url, tags] in enumerate(lines):
#           self.ext_infoid_dict[info_id] = {}
#           self.ext_infoid_dict[info_id]['no'] = i
#           self.ext_infoid_dict[info_id]['tags'] = tags.split(' ')
        
        self.imgnames = [ [line[0], line[1], line[2]] for line in lines if line[0] not in self.data]
        logger.info("{0} images haven't been tagged".format(len(self.imgnames)))

        self.we = gensim.models.word2vec.Word2Vec.load(self.args['params']['wv_model'])


    def writecfg(self):
        fullpath = os.path.join(self.args['tagdir'], 'tag.yaml')
        with open(fullpath, 'w') as f:
            yaml.dump(self.args, f)

        fullpath = os.path.join(self.args['tagdir'], 'img.lst')
        with open(fullpath, 'w') as f:
            for item in self.imglst:
                f.write('\t'.join(item)+'\n')


    def get_tags(self, info_ids):
        similar_tags = []
        for info_id in info_ids:
            similar_tags.append(self.data[info_id]['tags'])
        res_tags = self.weighted_tags(similar_tags)
        return res_tags


    def get_more_tags(self, info_ids):
#       import pdb; pdb.set_trace()
        res_tags = {}
        weight, decay, decay_step = 1/0.98, 0.98, 4
        n_closet_word = 4
        for i, info_id in enumerate(info_ids):
            if i % decay_step == 0:
                weight *= decay

            for raw_tag in self.data[info_id]['tags']:
                res_tags[raw_tag] = res_tags.setdefault(raw_tag, 0) + weight
                if raw_tag not in self.we.wv.vocab:
                    continue
                for sim_tag, cos_sim in self.we.wv.most_similar(raw_tag, topn=n_closet_word):
                    res_tags[sim_tag] = res_tags.setdefault(sim_tag, 0) + weight * cos_sim

            if len(res_tags) > 70:
                break # 越靠后的图片越不准确
        res_tags = sorted(res_tags.items(), key=lambda x:x[1], reverse=True)
        return res_tags


    def weighted_tags(self, similar_tags):
        tags_weight = {}
        n_similar = len(similar_tags)
        for i, tags in enumerate(similar_tags):
            for tag in tags:
                tags_weight[tag] = tags_weight.setdefault(tag, 0) + (n_similar - i)
#               if tag not in tags_weight:
#                   tags_weight[tag] = (n_similar - i)
#               else:
#                   tags_weight[tag] += (n_similar - i)
        tags_weights = sorted(tags_weight.items(), key=lambda x:x[1], reverse=True)
        return tags_weights

    
    def tagged_offline(self, n_thread=3):
        logger.info('===== tagged offline =====')
        vec_iter = self.idxer.extor.load()
        vecs = [vec for vec in vec_iter]
        vecs = np.concatenate(vecs, axis=0)

        ext_imglst_file = os.path.join(self.idxer.extor.args['vecdir'], 'img.lst')
        with open(ext_imglst_file, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n').split('\t') for line in lines]
        self.ext_infoid_dict = {}
        for i, [info_id, pic_url, tags] in enumerate(lines):
            self.ext_infoid_dict[info_id] = {}
            self.ext_infoid_dict[info_id]['no'] = i
            self.ext_infoid_dict[info_id]['tags'] = tags.split(' ')
        self.imgnames = [ [line[0], line[1], line[2]] for line in lines if line[0] not in self.data]
        logger.info("{0} images haven't been tagged".format(len(self.imgnames)))
        
        new_imgnames = []
        for item in self.imgnames:
            info_id, pic_url, tags = item
            tags = tags.split(' ')
            tags = self.tag_proc(vecs, info_id, tags)
            tags = ' '.join(tags)
            new_imgnames.append([info_id, pic_url, tags])
        self.imglst.extend(new_imgnames)
        self.args['n_feature'] = len(self.imglst)
        self.writecfg()


    def tag_proc(self, vecs, info_id, tags):
        new_tags = mapping(tags)
        new_tags = list(filter(lambda x:x not in unchosen_tags, new_tags))
        new_tags = list(filter(lambda x:is_en1(x) or is_zh1(x), new_tags))
        if len(new_tags) != 0:
            return new_tags
        
        vec = vecs[self.ext_infoid_dict[info_id]['no']].reshape(1,-1)
        D, I = self.idxer.search(vec, 20) # topk不可设置太大，可能噪声太多
        D = D[0].tolist()
        info_ids, _ = self.idxer.get_imginfo(I)
        for infoid in info_ids:
            new_tags.append(self.ext_infoid_dict[infoid]['tags'])
        new_tags = self.weighted_tags(new_tags)

        new_tags = [k for k, v in new_tags]
        new_tags = mapping(new_tags)
        new_tags = list(filter(lambda x:x not in unchosen_tags, new_tags))
        new_tags = list(filter(lambda x:is_en1(x) or is_zh1(x), new_tags))
        return new_tags[:6]    #出于同样的原因，如果全部返回new_tags，new_tags的后面标签会很不准确，影响线上检索效果
