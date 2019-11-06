import json
import string


def read_corpus(file_path, source='src', verbose=True):
    # return list of list of words
    corpus = json.load(open(file_path))
    review_texts = []
    for review in corpus:
        try:
            text = review["text"].lower()
        except KeyError:
            continue
        text = text.translate(str.maketrans('', '', string.punctuation))
        review_texts.append(text.split())
    return review_texts


def load_partial_state_dict(model, state_dict):
    model_state = model.state_dict()
    loaded_keys = []
    unloaded_keys = []
    unseen_keys = []
    for name, param in state_dict.items():
        if name not in model_state:
            if re.fullmatch(LINEAR_BACKCOMP_PATTERN_1, name) or re.fullmatch(LINEAR_BACKCOMP_PATTERN_2, name):
                # backwards compability from standard to potentially-factorized transformer
                name = name.split('.')
                name.insert(-1, "linear")
                name = ".".join(name)
            else:
                unloaded_keys.append(name)
                continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            model_state[name].copy_(param)
        except KeyError:
            unloaded_keys.append(name)
            continue
        except RuntimeError:
            print(param.shape)
            print(model_state[name].shape)
            unloaded_keys.append(name)
            continue
        loaded_keys.append(name)
    for name, param in model_state.items():
        if name not in loaded_keys:
            unseen_keys.append(name)
    if len(unseen_keys) > 0:
        print("{} params not found in file :".format(len(unseen_keys)), unseen_keys)
        print()
    if len(unloaded_keys) > 0:
        print("{} params in file not in model :".format(len(unloaded_keys)), unloaded_keys)
        print()
    if len(unseen_keys) == 0 and len(unloaded_keys) == 0:
        print("Model and file matching !")
        print()