from os import path as osp
from configuration import TaskConfig as tkconfig

data_folder = osp.join("..", "data")
task_folder = osp.join(data_folder, tkconfig.dataset)
vocab_path = osp.join(task_folder, "vocab.bin")
subwords_folder = osp.join(task_folder, "subwords")
subwords_data_path = osp.join(subwords_folder, "train.text.split.subwords.en")


def get_data_path(chunk, origin):
    assert chunk in ["train", "valid", "test"], "Passed {} but chunk must be one of train, valid, test".format(chunk)
    assert origin in ["src", "tgt"], "Passed {} but origin must be one of src, tgt".format(origin)

    if tkconfig.task == "sentiment":
        if origin == "src":
            return osp.join(data_folder, tkconfig.dataset, "{}.text.en".format(chunk))
        if origin == "tgt":
            return osp.join(data_folder, tkconfig.dataset, "{}.scores.npy".format(chunk))

    raise KeyError("Task {} not recognized".format(tkconfig.task))
