import pickle as pkl
import pandas as pd
from os.path import join
from functools import partial
import torch

PREFIXES = {"clf": {
    "2019": "3404067_clf-2019",
    "2016": "3403636_clf-2016"
}}
EXCLUDED_LANGS = ["basque", "uzbek"]

def get_clf_in_prefix(dataset, prefix=None):
    if not prefix: prefix = PREFIXES["clf"][dataset]
    file = join(prefix, "results/clf.pkl")
    with open(file, 'rb') as f:
        data = pkl.load(f)

    df = pd.DataFrame.from_records(data, coerce_float=True).transpose()
    df["best_model"] = df["best_model"].apply(partial(join, prefix))
    df = df.rename_axis(['timestamp']).reset_index(level=0)
    df["dataset, lang, version"] = dataset + " - " + df["lang"] + " - " + df["seed_id"].apply(str)
    df = df.sort_index().drop_duplicates(subset=("seed","dataset, lang, version"),keep='last')
    #print(df["dataset, lang, version"].unique().size / len(df["dataset, lang, version"]))
    df.index = df["dataset, lang, version"]
    return df["best_model"]


from os.path import exists, abspath, dirname, join

THIS_FOLDER = dirname(abspath(__file__))
DEFAULT_LOG_FOLDER = join(THIS_FOLDER, "..", "results")
DEFAULT_MODEL_FOLDER = join(THIS_FOLDER, "..", "models")

CLF_EXPE_TAG = "clf/{dataset}/{language}/{seed_id}"
RET_EXPE_TAG = "ret/{dataset}/{language}/{seed_id}"
RET_ANNC_EXPE_TAG = "ret-annc/{dataset}/{language}/{seed_id}"
_3COS_EXPE_TAG = "3cos{variant}/{dataset}/{language}/{seed_id}"

# === Model path ===
def get_clf_expe_tag(dataset, language, seed_id):
    return CLF_EXPE_TAG.format(dataset=dataset, language=language, seed_id=seed_id)
def get_ret_expe_tag(dataset, language, seed_id):
    return RET_EXPE_TAG.format(dataset=dataset, language=language, seed_id=seed_id)
def get_ret_annc_expe_tag(dataset, language, seed_id):
    return RET_ANNC_EXPE_TAG.format(dataset=dataset, language=language, seed_id=seed_id)
def get_3cos_expe_tag(variant, dataset, language, seed_id):
    return _3COS_EXPE_TAG.format(variant=variant, dataset=dataset, language=language, seed_id=seed_id)

def get_clf_model_path(dataset, language, seed_id):
    return join(DEFAULT_MODEL_FOLDER, get_clf_expe_tag(dataset, language, seed_id), f"model.pkl")
def get_ret_model_path(dataset, language, seed_id):
    return join(DEFAULT_MODEL_FOLDER, get_ret_expe_tag(dataset, language, seed_id), f"model.pkl")

def get_clf_report_path(dataset, language, seed_id):
    return join(DEFAULT_MODEL_FOLDER, get_clf_expe_tag(dataset, language, seed_id), f"summary.csv")
def get_ret_report_path(dataset, language, seed_id):
    return join(DEFAULT_MODEL_FOLDER, get_ret_expe_tag(dataset, language, seed_id), f"summary.csv")
def get_ret_annc_report_path(dataset, language, seed_id):
    return join(DEFAULT_MODEL_FOLDER, get_ret_annc_expe_tag(dataset, language, seed_id), f"summary.csv")
def get_3cos_report_path(variant, dataset, language, seed_id):
    return join(DEFAULT_MODEL_FOLDER, get_3cos_expe_tag(variant, dataset, language, seed_id), f"summary.csv")

def get_clf_fails_path(dataset, language, seed_id):
    return join(DEFAULT_LOG_FOLDER, get_clf_expe_tag(dataset, language, seed_id), f"fails.csv")
def get_ret_fails_path(dataset, language, seed_id):
    return join(DEFAULT_LOG_FOLDER, get_ret_expe_tag(dataset, language, seed_id), f"fails.csv")
def get_ret_annc_fails_path(dataset, language, seed_id):
    return join(DEFAULT_LOG_FOLDER, get_ret_annc_expe_tag(dataset, language, seed_id), f"fails.csv")
def get_3cos_fails_path(variant, dataset, language, seed_id):
    return join(DEFAULT_LOG_FOLDER, get_3cos_expe_tag(variant, dataset, language, seed_id), f"fails.csv")

def get_special_clf_model_path():
    pass
def get_special_ret_model_path():
    pass

# === Model loading from a path ===
def load_final_model(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)
def load_cnn_emb(path, map_location="cpu"):
    return load_final_model(path, map_location="cpu")["state_dict"]["cnn_emb"]
def load_ae_emb(path, map_location="cpu"):
    return load_final_model(path, map_location="cpu")["state_dict"]["ae_emb"]
def load_annc(path, map_location="cpu"):
    return load_final_model(path, map_location="cpu")["state_dict"]["annc"]
def load_annr(path, map_location="cpu"):
    return load_final_model(path, map_location="cpu")["state_dict"]["annr"]
def save_final_model(path, performance_dict=None, other_info=None, cnn_emb_state_dict=None, ae_emb_state_dict=None, annc_state_dict=None, annr_state_dict=None):
    torch.save({
        "state_dict": {
            "cnn_emb": cnn_emb_state_dict,
            "ae_emb": ae_emb_state_dict,
            "annc": annc_state_dict,
            "annr": annr_state_dict
        },
        "performance_dict": performance_dict,
        "other_info": other_info
    }, path)

