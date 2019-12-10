from os import path as osp

from configuration import EXPERIMENT_NAME

code_path = osp.dirname(osp.realpath(__file__))
project_path = osp.join(code_path, "..")
data_path = osp.join(project_path, "data")
senteval_data_path = osp.join(data_path, "senteval")
semeval_data_path = osp.join(data_path, "semeval")
experiment_path = osp.join(project_path, "experiments", EXPERIMENT_NAME)
results_path = osp.join(project_path, "results", EXPERIMENT_NAME)
direct_reload_path = osp.join(experiment_path, "SNLI")
