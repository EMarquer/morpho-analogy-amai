
try:
    from . import train_clf_glove_fasttext
    from ..utils.lightning import fix_gpu_args
except ImportError:
    import train_clf_glove_fasttext
    import sys, os
    sys.path.append(os.path.abspath(train_clf_glove_fasttext.IMPORTS_PATH))
    from utils.lightning import fix_gpu_args

if __name__ == "__main__":
    #for data_seed_id in range(5):
    data_seed_id = 0
    for model_seed_id in range(10):
        for model in train_clf_glove_fasttext.MODELS.keys():
            for post_emb in ["", " --post-emb"]:
                # train clf

                if "CNN+ANNc" not in model:

                    # argument parsing
                    str_args=f"-Vm {model_seed_id} -m {model} --skip --max_epochs 20 --gpus=-1 -t 5000 -b 512 {post_emb}" # 128
                    parser = train_clf_glove_fasttext.add_argparse_args()
                    args = parser.parse_args(str_args.split())
                    fix_gpu_args(args)

                    print(f"Running for `{str_args}`")
                    train_clf_glove_fasttext.main(args)

    
    for model_seed_id in range(5):
        for model in train_clf_glove_fasttext.MODELS.keys():
            for post_emb in ["", " --post-emb"]:
                # train clf

                # argument parsing
                str_args=f"-Vm {model_seed_id} -m {model} --skip --max_epochs 20 --gpus=-1 -t 5000 -b 512 {post_emb} --covered" # 128
                parser = train_clf_glove_fasttext.add_argparse_args()
                args = parser.parse_args(str_args.split())
                fix_gpu_args(args)

                print(f"Running for `{str_args}`")
                train_clf_glove_fasttext.main(args)