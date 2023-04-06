from os.path import exists, dirname, join
from os import mkdir
import json
import logging

logger = logging.getLogger(__name__)

THIS_DIR = dirname(__file__)
ROOT = join(THIS_DIR, "..")

# custom logging
CUSTOM_CONFIG_FILE_NAME = "siganalogies.cfg.json"
CUSTOM_CONFIG_FILE = None
LOADED_CONFIG = None
if (# we look, in order of priority
    exists(file_name:=CUSTOM_CONFIG_FILE_NAME) or #local
    exists(file_name:=join(ROOT, CUSTOM_CONFIG_FILE_NAME)) or #at base of packages
    exists(file_name:=join(THIS_DIR, CUSTOM_CONFIG_FILE_NAME))): #in siganalogies package
    CUSTOM_CONFIG_FILE = file_name
    try:
        with open(CUSTOM_CONFIG_FILE, "r") as f:
            LOADED_CONFIG = json.load(f)
        logger.info(f"siganalogies config file found at {file_name}, it contains the following keys:\n{list(LOADED_CONFIG.keys())}")
    except Exception:
        logger.info(f"siganalogies config file found at {file_name}, but loading failed")
else:
    logger.info(f"siganalogies config file not found at either {CUSTOM_CONFIG_FILE_NAME}, {join(ROOT, CUSTOM_CONFIG_FILE_NAME)}, or {join(THIS_DIR, CUSTOM_CONFIG_FILE_NAME)}")
def cfg(name):
    if LOADED_CONFIG and name in LOADED_CONFIG.keys():
        return LOADED_CONFIG[name]
    else:
        return None

AUTO_DOWNLOAD_SIG = cfg("AUTO_DOWNLOAD_SIG") if cfg("AUTO_DOWNLOAD_SIG") is not None else False

SERIALIZATION = cfg("SERIALIZATION") if cfg("SERIALIZATION") is not None else True
DOWNLOAD = cfg("DOWNLOAD") if cfg("DOWNLOAD") is not None else True
DOREL_URL = "https://dorel.univ-lorraine.fr/api/access/datafile/:persistentId?persistentId=doi:10.12763/MLCFIE"
DOREL_PREFIX = "doi:10.12763/MLCFIE/"
DOREL_JSON_DESCRIPTION_URL = "https://dorel.univ-lorraine.fr/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.12763/MLCFIE"
def dorel_json_description() -> dict:
    """Return a description of the dataset as JSON from DOREL."""
    import urllib.request as r
    with r.urlopen(DOREL_JSON_DESCRIPTION_URL) as url:
        data = json.load(url)
    return data
def dorel_pkl_url(dataset="2016", language="german", mode="train", word_encoder="none") -> str:
    """Return the file URL of the pickle file as JSON from DOREL."""
    from .encoders import encoder_as_string
    file_name = f"{language}-{mode}-{encoder_as_string(word_encoder)}.pkl"
    for file in dorel_json_description()["datasetVersion"]["files"]:
        if file.get("directoryLabel", "") == f"precomputed/{dataset}" and file["dataFile"]["filename"] == file_name:
            file_id = file["dataFile"]["persistentId"].split("/")[-1]
            return f"{DOREL_URL}/{file_id}"

DATASET_PATH = cfg("DATASET_PATH") or join(THIS_DIR, "precomputed")
if not exists(DATASET_PATH):
    mkdir(DATASET_PATH)

CUSTOM_SERIALIZATION_PATH = cfg("CUSTOM_SERIALIZATION_PATH") or join(DATASET_PATH, "custom")
if not exists(CUSTOM_SERIALIZATION_PATH):
    mkdir(CUSTOM_SERIALIZATION_PATH)

SIG2016_DATASET_PATH = cfg("SIG2016_DATASET_PATH") or join(THIS_DIR, "sigmorphon2016/data/")
SIG2016_SERIALIZATION_PATH = cfg("SIG2016_SERIALIZATION_PATH") or join(DATASET_PATH, "2016")
if not exists(SIG2016_SERIALIZATION_PATH):
    mkdir(SIG2016_SERIALIZATION_PATH)
SIG2016_LANGUAGES = ["arabic", "finnish", "georgian", "german", "hungarian", "japanese", "maltese", "navajo", "russian", "spanish", "turkish"]
SIG2016_LANGUAGES_SHORT = ["ar", "fi", "ka", "de", "hu", "mt", "nv", "ru", "es", "tr"] # for BPEmb subword embeddings
SIG2016_MODES = ["train", "dev", "test"] # "test-covered" is not used because it is not exactly the same format as the others

SIG2019_DATASET_PATH = cfg("SIG2019_DATASET_PATH") or join(THIS_DIR, "sigmorphon2019/task1")
SIG2019_SERIALIZATION_PATH = cfg("SIG2019_SERIALIZATION_PATH") or join(DATASET_PATH, "2019")
if not exists(SIG2019_SERIALIZATION_PATH):
    mkdir(SIG2019_SERIALIZATION_PATH)
SIG2019_FOLDERS = [
    "adyghe--kabardian", "albanian--breton", "arabic--classical-syriac", "arabic--maltese", "arabic--turkmen", "armenian--kabardian",
    "asturian--occitan", "bashkir--azeri", "bashkir--crimean-tatar", "bashkir--kazakh", "bashkir--khakas", "bashkir--tatar", "bashkir--turkmen",
    "basque--kashubian", "belarusian--old-irish", "bengali--greek", "bulgarian--old-church-slavonic", "czech--kashubian", "czech--latin",
    "danish--middle-high-german", "danish--middle-low-german", "danish--north-frisian", "danish--west-frisian", "danish--yiddish",
    "dutch--middle-high-german", "dutch--middle-low-german", "dutch--north-frisian", "dutch--west-frisian", "dutch--yiddish", "english--murrinhpatha",
    "english--north-frisian", "english--west-frisian", "estonian--ingrian", "estonian--karelian", "estonian--livonian", "estonian--votic",
    "finnish--ingrian", "finnish--karelian", "finnish--livonian", "finnish--votic", "french--occitan", "german--middle-high-german",
    "german--middle-low-german", "german--yiddish", "greek--bengali", "hebrew--classical-syriac", "hebrew--maltese", "hindi--bengali",
    "hungarian--ingrian", "hungarian--karelian", "hungarian--livonian", "hungarian--votic", "irish--breton", "irish--cornish",
    "irish--old-irish", "irish--scottish-gaelic", "italian--friulian", "italian--ladin", "italian--maltese", "italian--neapolitan",
    "kannada--telugu", "kurmanji--sorani", "latin--czech", "latvian--lithuanian", "latvian--scottish-gaelic", "persian--azeri",
    "persian--pashto", "polish--kashubian", "polish--old-church-slavonic", "portuguese--russian", "romanian--latin", "russian--old-church-slavonic",
    "russian--portuguese", "sanskrit--bengali", "sanskrit--pashto", "slovak--kashubian", "slovene--old-saxon", "sorani--irish", "spanish--friulian",
    "spanish--occitan", "swahili--quechua", "turkish--azeri", "turkish--crimean-tatar", "turkish--kazakh", "turkish--khakas", "turkish--tatar",
    "turkish--turkmen", "urdu--bengali", "urdu--old-english", "uzbek--azeri", "uzbek--crimean-tatar", "uzbek--kazakh", "uzbek--khakas", "uzbek--tatar",
    "uzbek--turkmen", "welsh--breton", "welsh--cornish", "welsh--old-irish", "welsh--scottish-gaelic", "zulu--swahili"]
SIG2019_HIGH = [
    "adyghe", "albanian", "arabic", "armenian", "asturian", "bashkir", "basque", "belarusian", "bengali", "bulgarian", "czech", "danish", "dutch",
    "english", "estonian", "finnish", "french", "german", "greek", "hebrew", "hindi", "hungarian", "irish", "italian", "kannada", "kurmanji",
    "latin", "latvian", "persian", "polish", "portuguese", "romanian", "russian", "sanskrit", "slovak", "slovene", "sorani", "spanish",
    "swahili", "turkish", "urdu", "uzbek", "welsh", "zulu"]
SIG2019_LOW = [
    "azeri", "bengali", "breton", "classical-syriac", "cornish", "crimean-tatar", "czech", "friulian", "greek", "ingrian", "irish", "kabardian", "karelian",
    "kashubian", "kazakh", "khakas", "ladin", "latin", "lithuanian", "livonian", "maltese", "middle-high-german", "middle-low-german", "murrinhpatha", "neapolitan", "north-frisian", "occitan", "old-church-slavonic", "old-english", "old-irish", "old-saxon", "pashto", "portuguese", "quechua", "russian", "scottish-gaelic", "sorani", "swahili", "tatar", "telugu", "turkmen", "votic", "west-frisian", "yiddish"]
SIG2019_LANGUAGES = sorted(list(set(SIG2019_HIGH + SIG2019_LOW)))
SIG2019_HIGH_LOW_PAIRS = [
    ("adyghe", "kabardian"), ("albanian", "breton"), ("arabic", "classical-syriac"), ("arabic", "maltese"), ("arabic", "turkmen"), ("armenian", "kabardian"),
    ("asturian", "occitan"), ("bashkir", "azeri"), ("bashkir", "crimean-tatar"), ("bashkir", "kazakh"), ("bashkir", "khakas"), ("bashkir", "tatar"), ("bashkir", "turkmen"), ("basque", "kashubian"), ("belarusian", "old-irish"), ("bengali", "greek"), ("bulgarian", "old-church-slavonic"), ("czech", "kashubian"), ("czech", "latin"), ("danish", "middle-high-german"), ("danish", "middle-low-german"), ("danish", "north-frisian"), ("danish", "west-frisian"), ("danish", "yiddish"), ("dutch", "middle-high-german"), ("dutch", "middle-low-german"), ("dutch", "north-frisian"), ("dutch", "west-frisian"), ("dutch", "yiddish"), ("english", "murrinhpatha"), ("english", "north-frisian"), ("english", "west-frisian"), ("estonian", "ingrian"), ("estonian", "karelian"), ("estonian", "livonian"), ("estonian", "votic"), ("finnish", "ingrian"), ("finnish", "karelian"), ("finnish", "livonian"), ("finnish", "votic"), ("french", "occitan"), ("german", "middle-high-german"), ("german", "middle-low-german"), ("german", "yiddish"), ("greek", "bengali"), ("hebrew", "classical-syriac"), ("hebrew", "maltese"), ("hindi", "bengali"), ("hungarian", "ingrian"), ("hungarian", "karelian"), ("hungarian", "livonian"), ("hungarian", "votic"), ("irish", "breton"), ("irish", "cornish"), ("irish", "old-irish"), ("irish", "scottish-gaelic"), ("italian", "friulian"), ("italian", "ladin"), ("italian", "maltese"), ("italian", "neapolitan"), ("kannada", "telugu"), ("kurmanji", "sorani"), ("latin", "czech"), ("latvian", "lithuanian"), ("latvian", "scottish-gaelic"), ("persian", "azeri"), ("persian", "pashto"), ("polish", "kashubian"), ("polish", "old-church-slavonic"), ("portuguese", "russian"), ("romanian", "latin"), ("russian", "old-church-slavonic"), ("russian", "portuguese"), ("sanskrit", "bengali"), ("sanskrit", "pashto"), ("slovak", "kashubian"), ("slovene", "old-saxon"), ("sorani", "irish"), ("spanish", "friulian"), ("spanish", "occitan"), ("swahili", "quechua"), ("turkish", "azeri"), ("turkish", "crimean-tatar"), ("turkish", "kazakh"), ("turkish", "khakas"), ("turkish", "tatar"), ("turkish", "turkmen"), ("urdu", "bengali"), ("urdu", "old-english"), ("uzbek", "azeri"), ("uzbek", "crimean-tatar"), ("uzbek", "kazakh"), ("uzbek", "khakas"), ("uzbek", "tatar"), ("uzbek", "turkmen"), ("welsh", "breton"), ("welsh", "cornish"), ("welsh", "old-irish"), ("welsh", "scottish-gaelic"), ("zulu", "swahili")]
SIG2019_HIGH_MODES = ["train-high"]
SIG2019_LOW_MODES = ["train-low", "test", "dev"] # "test-covered" is not used because it is not exactly the same format as the others

#print(set(SIG2019_HIGH).intersection(set(SIG2019_LOW)))
JBATS_USE_KANJI = True