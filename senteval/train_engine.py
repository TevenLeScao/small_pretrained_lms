from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLI
from senteval.emocontext import EmoContext
from senteval.hateval import HatEval
from senteval.sentiment import SentimentAnalysis
from senteval.permutation_detection import PermutationDetection
from senteval.trec import TRECEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.mrpc import MRPCEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.sst import SSTEval
from senteval.rank import ImageCaptionRetrievalEval
from senteval.probing import *

from utils.helpers import dotdict, makedirs
from models.structure import *
from models.sentence_encoders import SentenceEncoder
import paths


class TrainEngine(object):
    def __init__(self, params, prepare=None):
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
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16',
                           'Length', 'WordContent', 'Depth', 'TopConstituents',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion', 'EmoContext', 'HatEval', 'Sentiment', 'Permutation']

    def train(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if not hasattr(self, 'results'):
            self.results = []
        if (isinstance(name, list)):
            for x in name:
                self.train(x)
            return
        else:
            assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'SNLI':
            self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
            self.trainer = SNLI(self.params.task_path, seed=self.params.seed)

        if name == "EmoContext":
            self.params.task_path = self.params.semeval_path + '/downstream/{}'.format(name)
            self.trainer = EmoContext(self.params.task_path, seed=self.params.seed)

        if name == "HatEval":
            self.params.task_path = self.params.semeval_path + '/downstream/{}'.format(name)
            self.trainer = HatEval(self.params.task_path, seed=self.params.seed)

        if name == "Sentiment":
            self.params.task_path = self.params.others_path + '/downstream/{}'.format(name)
            self.trainer = SentimentAnalysis(self.params.task_path, seed=self.params.seed)

        if name == "Permutation":
            self.params.task_path = self.params.others_path + '/downstream/{}'.format(name)
            self.trainer = PermutationDetection(self.params.task_path, seed=self.params.seed)
        # TODO: convert other tasks
        # if name == 'CR':
        #     self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
        #     self.trainer = CREval(self.params.task_path, seed=self.params.seed)
        # elif name == 'MR':
        #     self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
        #     self.trainer = MREval(self.params.task_path, seed=self.params.seed)
        # elif name == 'MPQA':
        #     self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
        #     self.trainer = MPQAEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'SUBJ':
        #     self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
        #     self.trainer = SUBJEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'SST2':
        #     self.params.task_path = self.params.base_path + '/downstream/SST/binary'
        #     self.trainer = SSTEval(self.params.task_path, nclasses=2, seed=self.params.seed)
        # elif name == 'SST5':
        #     self.params.task_path = self.params.base_path + '/downstream/SST/fine'
        #     self.trainer = SSTEval(self.params.task_path, nclasses=5, seed=self.params.seed)
        # elif name == 'TREC':
        #     self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
        #     self.trainer = TRECEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'MRPC':
        #     self.params.task_path = self.params.base_path + '/downstream/{}'.format(name)
        #     self.trainer = MRPCEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'SICKRelatedness':
        #     self.params.task_path = self.params.base_path + '/downstream/SICK'
        #     self.trainer = SICKRelatednessEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'STSBenchmark':
        #     self.params.task_path = self.params.base_path + '/downstream/STS/STSBenchmark'
        #     self.trainer = STSBenchmarkEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'SICKEntailment':
        #     self.params.task_path = self.params.base_path + '/downstream/SICK'
        #     self.trainer = SICKEntailmentEval(self.params.task_path, seed=self.params.seed)
        # elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        #     self.params.task_path = self.params.base_path + '/downstream/STS/{}'.format(name + '-en-test')
        #     self.trainer = eval(name + 'Eval')(self.params.task_path, seed=self.params.seed)
        elif name == 'ImageCaptionRetrieval':
            self.params.task_path = self.params.base_path + '/downstream/COCO'
            self.trainer = ImageCaptionRetrievalEval(self.params.task_path, seed=self.params.seed)

        # Probing Tasks
        # elif name == 'Length':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = LengthEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'WordContent':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = WordContentEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'Depth':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = DepthEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'TopConstituents':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = TopConstituentsEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'BigramShift':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = BigramShiftEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'Tense':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = TenseEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'SubjNumber':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = SubjNumberEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'ObjNumber':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = ObjNumberEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'OddManOut':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = OddManOutEval(self.params.task_path, seed=self.params.seed)
        # elif name == 'CoordinationInversion':
        #     self.params.task_path = self.params.base_path + '/probing'
        #     self.trainer = CoordinationInversionEval(self.params.task_path, seed=self.params.seed)

        self.params.current_task = name
        self.params.current_xp_folder = os.path.join(paths.experiment_path, name)
        makedirs(self.params.current_xp_folder)
        self.trainer.do_train_prepare(self.params, self.prepare)

        self.results = self.trainer.train(self.params)

        return self.results