# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

import json

import paths

from utils.helpers import dotdict, makedirs
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLI
from senteval.trec import TRECEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.mrpc import MRPCEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.sst import SSTEval
from senteval.rank import ImageCaptionRetrievalEval
from senteval.probing import *
from senteval.emocontext import EmoContext
from senteval.hateval import HatEval


class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16',
                           'Length', 'WordContent', 'Depth', 'TopConstituents',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion', 'EmoContext',
                           'HatEval']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'CR':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.evaluation = CREval(self.params.task_path, seed=self.params.seed)
        elif name == 'MR':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.evaluation = MREval(self.params.task_path, seed=self.params.seed)
        elif name == 'MPQA':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.evaluation = MPQAEval(self.params.task_path, seed=self.params.seed)
        elif name == 'SUBJ':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.evaluation = SUBJEval(self.params.task_path, seed=self.params.seed)
        elif name == 'SST2':
            self.params.task_path = self.params.base_path + '/downstream/SST/binary'
            self.evaluation = SSTEval(self.params.task_path, nclasses=2, seed=self.params.seed)
        elif name == 'SST5':
            self.params.task_path = self.params.base_path + '/downstream/SST/fine'
            self.evaluation = SSTEval(self.params.task_path, nclasses=5, seed=self.params.seed)
        elif name == 'TREC':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.evaluation = TRECEval(self.params.task_path, seed=self.params.seed)
        elif name == 'MRPC':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.evaluation = MRPCEval(self.params.task_path, seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.params.task_path = self.params.base_path + '/downstream/SICK'
            self.evaluation = SICKRelatednessEval(self.params.task_path, seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.params.task_path = self.params.base_path + '/downstream/STS/STSBenchmark'
            self.evaluation = STSBenchmarkEval(self.params.task_path, seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.params.task_path = self.params.base_path + '/downstream/SICK'
            self.evaluation = SICKEntailmentEval(self.params.task_path, seed=self.params.seed)
        elif name == 'SNLI':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.evaluation = SNLI(self.params.task_path, seed=self.params.seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            self.params.task_path = self.params.base_path + '/downstream/STS/{}'.format(name + '-en-test')
            self.evaluation = eval(name + 'Eval')(self.params.task_path, seed=self.params.seed)
        elif name == 'ImageCaptionRetrieval':
            self.params.task_path = self.params.base_path + '/downstream/COCO'
            self.evaluation = ImageCaptionRetrievalEval(self.params.task_path, seed=self.params.seed)
        elif name == 'EmoContext':
            self.params.task_path = self.params.base_path + '/downstream/EmoContext'
            self.evaluation = EmoContext(self.params.task_path, seed=self.params.seed)
        elif name == 'HatEval':
            self.params.task_path = self.params.base_path + '/downstream/HatEval'
            self.evaluation = HatEval(self.params.task_path, seed=self.params.seed)

        # Probing Tasks
        elif name == 'Length':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = LengthEval(self.params.task_path, seed=self.params.seed)
        elif name == 'WordContent':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = WordContentEval(self.params.task_path, seed=self.params.seed)
        elif name == 'Depth':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = DepthEval(self.params.task_path, seed=self.params.seed)
        elif name == 'TopConstituents':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = TopConstituentsEval(self.params.task_path, seed=self.params.seed)
        elif name == 'BigramShift':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = BigramShiftEval(self.params.task_path, seed=self.params.seed)
        elif name == 'Tense':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = TenseEval(self.params.task_path, seed=self.params.seed)
        elif name == 'SubjNumber':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = SubjNumberEval(self.params.task_path, seed=self.params.seed)
        elif name == 'ObjNumber':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = ObjNumberEval(self.params.task_path, seed=self.params.seed)
        elif name == 'OddManOut':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = OddManOutEval(self.params.task_path, seed=self.params.seed)
        elif name == 'CoordinationInversion':
            self.params.task_path = self.params.base_path + '/probing'
            self.evaluation = CoordinationInversionEval(self.params.task_path, seed=self.params.seed)

        self.params.current_task = name
        self.params.current_xp_folder = os.path.join(paths.results_path, name)
        print(self.params.current_xp_folder)
        makedirs(self.params.current_xp_folder)
        output_json_file = os.path.join(self.params.current_xp_folder, "results.json")

        self.evaluation.do_prepare(self.params, self.prepare)
        self.results = self.evaluation.run(self.params, self.batcher)
        json.dump(self.results, open(output_json_file, "w"), indent=2, ensure_ascii=False)

        return self.results
