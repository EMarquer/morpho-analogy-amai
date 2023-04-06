
# Autoencoder average performance >= 80%
# Bashkir    Sigmorphon 2019                                                   0.80380
# Slovene    Sigmorphon 2019                                                   0.80948
# Adyghe     Sigmorphon 2019                                                   0.81904
# Swahili    Sigmorphon 2019                                                   0.81936
# Zulu       Sigmorphon 2019                                                   0.81964
# Slovak     Sigmorphon 2019                                                   0.82620
# Turkish    Sigmorphon 2016 + Japanese Bigger Analogy Test Set                0.83032
# Hungarian  Sigmorphon 2016 + Japanese Bigger Analogy Test Set                0.83572
# Sanskrit   Sigmorphon 2019                                                   0.83828
# Russian    Sigmorphon 2016 + Japanese Bigger Analogy Test Set                0.84704 --> produces bug when testing
# Georgian   Sigmorphon 2016 + Japanese Bigger Analogy Test Set                0.87056
# Welsh      Sigmorphon 2019                                                   0.87620
# Hebrew     Sigmorphon 2019                                                   0.91160

try:
    from .symbolic_baseline import _3cos
    from .utils.lightning import fix_gpu_args
except ImportError:
    import symbolic_baseline._3cos as _3cos
    import train_ret
    from utils.lightning import fix_gpu_args
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH


if __name__ == "__main__":
    #language_dataset = [(lang, "2016") for lang in SIG2016_LANGUAGES] + [(lang, "2019") for lang in SIG2019_HIGH]
    language_dataset = [(lang, "2016") for lang in ["turkish", "hungarian", "georgian"]] # "russian", 
    language_dataset += [(lang, "2019") for lang in ["arabic", "english", "french", "portuguese", "bashkir", "slovene", "adyghe", "swahili", "zulu", "slovak", "sanskrit", "welsh", "hebrew"]]

    #for data_seed_id in range(5):
    data_seed_id = 0
    for model_seed_id in range(10):
        for language, dataset in language_dataset:
            # train clf

            # argument parsing
            str_args=f"-Vm {model_seed_id} -l {language} -d {dataset} --skip --max_epochs 20 --gpus=-1 -t 5000 -b 256 --variant 3CosMul" # 128
            parser = _3cos.add_argparse_args()
            args = parser.parse_args(str_args.split())
            fix_gpu_args(args)

            print(f"Running for `{str_args}`")
            _3cos.main(args)