import configuration

vocab = "../data/vocab.bin"

def get_data_path(chunk, mode):
    prefix = data_aligned_folder
    if mode == "tgt":
        suffix = language
    else:
        assert mode == "src"
        suffix = "en"
    return prefix + chunk + ".en-{}.".format(language) + suffix