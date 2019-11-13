from os import path as osp
from configuration import TaskConfig as tkconfig

data_folder = osp.join("..", "data")
vocab_path = "../data/{}/vocab.bin".format(tkconfig.dataset)


def get_data_path(chunk, origin):
    assert chunk in ["train", "valid", "test"], "Passed {} but chunk must be one of train, valid, test".format(chunk)
    assert origin in ["src", "tgt"], "Passed {} but origin must be one of src, tgt".format(origin)

    if tkconfig.task == "sentiment":
        if origin == "src":
            return osp.join(data_folder, tkconfig.dataset, "{}.text.en".format(chunk))
        if origin == "tgt":
            return osp.join(data_folder, tkconfig.dataset, "{}.scores.npy".format(chunk))

    raise KeyError("Task {} not recognized".format(tkconfig.task))
