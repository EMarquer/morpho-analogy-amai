import logging
logger = logging.getLogger("")#__name__)
logger.setLevel(logging.INFO)

# configure logging at the root level of lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the loggers
logger.addHandler(ch)

import torch
import torch.nn as nn
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from datetime import datetime
from argparse import ArgumentParser

import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PARENT_DIR)
#sys.path.insert(0, PARENT_DIR+"/..")

try:
    from ..cnn_embeddings import CNNEmbedding
    from ..siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich
    from ..utils import prepare_data, precision_sak_ap_rr, embeddings_voc
    from ..utils.logger import to_csv
    from ..utils.lightning import get_trainer_kwargs, fix_gpu_args
    from ..utils.ckpt import get_clf_model_path, get_3cos_report_path, get_3cos_expe_tag, get_3cos_fails_path, DEFAULT_LOG_FOLDER, save_final_model, load_cnn_emb
except ImportError:
    from cnn_embeddings import CNNEmbedding
    from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich
    from utils import prepare_data, precision_sak_ap_rr, embeddings_voc
    from utils.logger import to_csv
    from utils.lightning import get_trainer_kwargs, fix_gpu_args
    from utils.ckpt import get_clf_model_path, get_3cos_report_path, get_3cos_expe_tag, get_3cos_fails_path, DEFAULT_LOG_FOLDER, save_final_model, load_cnn_emb

import os
os.environ['PYTHONHASHSEED'] = str(42)
seed_everything(42, workers=True)

RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
VERSION = 1.0

class CosLightning(pl.LightningModule):
    def __init__(self, char_emb_size, encoder, variant = "3CosAdd", report_path="", fails_path=""):
        super().__init__()
        self.save_hyperparameters()
        self.emb = CNNEmbedding(voc_size=len(encoder.id_to_char), char_emb_size=char_emb_size)
        self.encoder = encoder
        self.variant = variant
        
        self.voc = None

        self.report_path = report_path
        self.fails_path = fails_path
        self.extra_info = dict()
        self.test_performance = dict()
        self.force_eval_on_cpu = False

    def test_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a_e = self.emb(a)
        b_e = self.emb(b)
        c_e = self.emb(c)
        d_e = self.emb(d)

        scores = []
        fails = []

        # positive example, target is 1
        for (a_, a_e_), (b_, b_e_), (c_, c_e_), (d_, d_e_) in enrich((a, a_e), (b, b_e), (c, c_e), (d, d_e)):
            #with elapsed_timer() as t:
            p, sak, r, rr, pred_w, tgt_w = precision_sak_ap_rr((a_e_, b_e_, c_e_), d_e_, self.voc, k=[3,5,10], strategy=self.variant)

            mask = p < 1
            indices = torch.arange(a.size(0), device=p.device)[mask]
            for i in indices:
                fails.append({
                        "A": self.encoder.decode(a_[i], pad_char=''),
                        "B": self.encoder.decode(b_[i], pad_char=''),
                        "C": self.encoder.decode(c_[i], pad_char=''),
                        "actual D": self.encoder.decode(d_[i], pad_char=''),
                        "predicted D": pred_w[i],
                        "target D": tgt_w[i],
                    })
            scores.append({
                'precision': p,
                'success@3': sak[0],
                'success@5': sak[1],
                'success@10': sak[2],
                'rr': rr
            })

        return {
            "scores": scores,
            "fails": fails
        }

    def test_epoch_end(self, outputs):
        gathered = self.all_gather(outputs)

        # When logging only on rank 0, don't forget to add ``rank_zero_only=True`` to avoid deadlocks on synchronization.
        if self.trainer.is_global_zero:
            scores = []
            fails = []
            for gathered_ in gathered:
                scores.extend(gathered_["scores"])
                fails.extend(gathered_["fails"])

            m_precision = torch.mean(torch.cat([score_dict["precision"] for score_dict in scores], dim=-1))
            m_sak3 = torch.mean(torch.cat([score_dict["success@3"] for score_dict in scores], dim=-1))
            m_sak5 = torch.mean(torch.cat([score_dict["success@5"] for score_dict in scores], dim=-1))
            m_sak10 = torch.mean(torch.cat([score_dict["success@10"] for score_dict in scores], dim=-1))
            m_rr = torch.mean(torch.cat([score_dict["rr"] for score_dict in scores], dim=-1))

            self.test_performance = {"precision": m_precision.item(), "success@3": m_sak3.item(), "success@5": m_sak5.item(), "success@10": m_sak10.item(), "mrr": m_rr.item(), **self.extra_info}
            print(self.test_performance)
            #print(fails)
            to_csv(self.report_path, self.test_performance)
            to_csv(self.fails_path, fails)
            #self.log("my_reduced_metric", mean, rank_zero_only=True)


def load_emb(args, target_module: nn.Module):
    if not args.transfer:
        return
    if args.transfer == "auto":
        model_path = get_clf_model_path(args.dataset, args.language, args.model_seed_id)
    else:
        model_path = args.transfer
    state_dict = load_cnn_emb(model_path)
    target_module.load_state_dict(state_dict)

def main(args):
    variant_tag = "add" if args.variant == "3CosAdd" else "mul"
    report_path = get_3cos_report_path(variant_tag, args.dataset, args.language, args.model_seed_id)
    fails_path = get_3cos_fails_path(variant_tag, args.dataset, args.language, args.model_seed_id)
    expe_tag = get_3cos_expe_tag(variant_tag, args.dataset, args.language, args.model_seed_id)
    model_seed = RANDOM_SEEDS[args.model_seed_id]
    if args.skip and os.path.exists(report_path):
        print(f"Report {report_path} exists, aborting")
        return
    
    fix_gpu_args(args)

    if hasattr(args, 'deterministic') and args.deterministic:
        logger.warning("Determinism (--deterministic True) does not guarantee reproducible results when changing the number of processes.")

    train_loader, val_loader, test_loader, encoder = prepare_data(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=RANDOM_SEEDS[args.model_seed_id])

    # --- Define models ---
    char_emb_size = 64
    nn = CosLightning(char_emb_size=char_emb_size,
                      encoder=encoder,
                      variant=args.variant,
                      report_path=report_path,
                      fails_path=fails_path)
    load_emb(args, nn.emb)

    with torch.no_grad():
        with embeddings_voc(nn.emb, train_loader.dataset.dataset, test_loader.dataset.dataset,distributed_barier=False) as voc:
            trainer = pl.Trainer.from_argparse_args(args,
                logger = None,
                **get_trainer_kwargs(args)#, find_unused_parameters=True)
            )
            nn.voc = voc
            nn.voc.vectors = nn.voc.vectors.to(nn.device)
            nn.extra_info = {
                "timestamp": datetime.now().timestamp(),
                "seed": RANDOM_SEEDS[args.model_seed_id],
                "seed_id": args.model_seed_id,
                "lang": args.language,
                "dataset": args.dataset,
                "variant": args.variant}
            seed_everything(RANDOM_SEEDS[args.model_seed_id], workers=True)
            try:
                trainer.test(nn, dataloaders=test_loader)
            except RuntimeError as e: # Automatically falback to cpu if vocabulary is too large
                if 'out of memory' not in str(e).lower():
                    raise RuntimeError(e)
                nn.force_eval_on_cpu = True
                nn = nn.cpu()
                nn.voc.vectors = nn.voc.vectors.cpu()
                trainer.test(nn, dataloaders=test_loader)
            nn.voc = None


def add_argparse_args(parser=None, return_all=False):
    # argument parsing
    if parser is None:
        parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--variant', '-M', type=str, default="3CosAdd", choices=["3CosAdd", "3CosMul"], help='The model to use.')
    model_parser.add_argument('--model_seed_id', '-Vm', type=int, default=0, help='The model seed.')

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch-size', '-b', type=int, default=512, help='Batch size.')
    dataset_parser.add_argument('--transfer', '-T', type=str, default="auto", help='The path of the model to load ("auto" to select the correct reference classifier).')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already.', action='store_true')

    if return_all:
        return parser, dataset_parser, model_parser
    else:
        return parser

if __name__ == '__main__':
    # argument parsing
    parser = add_argparse_args()

    args = parser.parse_args()
    fix_gpu_args(args)

    main(args)

