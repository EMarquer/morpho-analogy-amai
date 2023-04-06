
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
    from . import train_annr_gen
except ImportError:
    import train_annr_gen
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH


if __name__ == "__main__":
    from argparse import ArgumentParser

    #language_dataset = [(lang, "2016") for lang in SIG2016_LANGUAGES] + [(lang, "2019") for lang in SIG2019_HIGH]
    language_dataset = [(lang, "2016") for lang in ["turkish", "hungarian", "georgian"]] # "russian", 
    language_dataset += [(lang, "2019") for lang in ["arabic", "english", "french", "portuguese", "bashkir", "slovene", "adyghe", "swahili", "zulu", "slovak", "sanskrit", "welsh", "hebrew"]]

    #for data_seed_id in range(5):
    data_seed_id = 0
    for model_seed_id in range(10):
            for language, dataset in language_dataset:
                str_args=f"-Vm {model_seed_id} -Vs {data_seed_id} -l {language} -d {dataset} --skip --max_epochs 50 --gpus=-1 -t 5000"
                print(f"Running for `{str_args}`")

                # argument parsing
                parser = ArgumentParser()
                parser, dataset_parser, seed_parser = train_annr_gen.add_argparse_args(parser)

                args = parser.parse_args(str_args.split())

                # handle version details
                train_annr_gen.fix_gpu_args(args)

                train_annr_gen.main(args)