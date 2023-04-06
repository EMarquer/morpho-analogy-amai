import logging
from math import ceil
from packaging import version
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
from torch.nn.functional import mse_loss
import torchtext.vocab as vocab
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from argparse import ArgumentParser

try:
    from .annr import ANNr, AnalogyRegressionLoss
    from .annc import ANNc
    from .cnn_embeddings import CNNEmbedding
    from .siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich
    from .utils import prepare_data, embeddings_voc
    from .utils.logger import to_csv
    from .utils.lightning import get_trainer_kwargs, fix_gpu_args
    from .utils.ckpt import get_clf_model_path, get_ret_annc_report_path, get_ret_annc_expe_tag, get_ret_annc_fails_path, DEFAULT_LOG_FOLDER, save_final_model
except ImportError:
    from annr import ANNr, AnalogyRegressionLoss
    from annc import ANNc
    from cnn_embeddings import CNNEmbedding
    from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich
    from utils import prepare_data, embeddings_voc
    from utils.logger import to_csv
    from utils.lightning import get_trainer_kwargs, fix_gpu_args
    from utils.ckpt import get_clf_model_path, get_ret_annc_report_path, get_ret_annc_expe_tag, get_ret_annc_fails_path, DEFAULT_LOG_FOLDER, save_final_model, load_annc, load_cnn_emb

import os
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:128"
seed_everything(42, workers=True)

torch.cuda.memory.set_per_process_memory_fraction(0.9)

RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
VERSION = 1.0


class RetANNcLightning(pl.LightningModule):
    def __init__(self, char_emb_size, encoder, filters = 128, report_path="", fails_path=""):
        super().__init__()
        self.save_hyperparameters()
        self.emb = CNNEmbedding(voc_size=len(encoder.id_to_char), char_emb_size=char_emb_size)
        self.clf = ANNc(emb_size=self.emb.get_emb_size(), filters = filters)
        self.encoder = encoder
        
        self.voc = None
        self.voc_vectors_expanded = None
        
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
            p, sak, r, rr, pred_w, tgt_w = self.precision_sak_ap_rr(a_e_, b_e_, c_e_, d_e_, self.encoder.decode(d_, pad_char='', eos_char='', bos_char=''), k=[3,5,10])

            mask = p < 1
            if isinstance(mask, bool):
                if mask:
                    fails.append({
                            "A": self.encoder.decode(a_, pad_char=''),
                            "B": self.encoder.decode(b_, pad_char=''),
                            "C": self.encoder.decode(c_, pad_char=''),
                            "actual D": self.encoder.decode(d_, pad_char=''),
                            "predicted D": pred_w,
                            "target D": tgt_w,
                        })
            else:
                indices = torch.arange(a.size(0), device=p.device)[mask]
                for i in indices:
                    fails.append({
                            "A": self.encoder.decode(a_[i], pad_char=''),
                            "B": self.encoder.decode(b_[i], pad_char=''),
                            "C": self.encoder.decode(c_[i], pad_char=''),
                            "actual D": self.encoder.decode(d_[i], pad_char=''),
                            "predicted D": pred_w,
                            "target D": tgt_w,
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
            #append_pkl(self.common_save_path, row)
            to_csv(self.report_path, self.test_performance)
            to_csv(self.fails_path, fails)
            #self.log("my_reduced_metric", mean, rank_zero_only=True)

    def closest_clf(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, ranks=False):
        if a.dim() == 1:
            # scores = self.clf(
            #     a, b, c, self.voc.vectors.to(a.device).unsqueeze(0)
            # )
        
            # if ranks:
            #     idcs = scores.argsort(descending=True, dim=-1)
            #     return self.voc.itos[idcs[0]], idcs
            # else:
            #     idx = scores.argmax(dim=-1)
            #     return self.voc.itos[idx], idx
            raise NotImplementedError()
        
        elif  a.dim() > 2:
            raise NotImplementedError()
            if ranks:
                idcs = scores.argsort(descending=True, dim=-1)
                return [self.voc.itos[idcs_[0]] for idcs_ in idcs], idcs
            else:
                idx = scores.argmax(dim=-1)
                return [self.voc.itos[idx_] for idx_ in idx], idx
        
        else:
            scores = self.clf(
                a.expand(self.voc.vectors.size()), b.expand(self.voc.vectors.size()), c.expand(self.voc.vectors.size()), self.voc.vectors.to(a.device)
            ).squeeze(-1)

            if ranks:
                idcs = scores.argsort(descending=True, dim=-1)
                return self.voc.itos[idcs[0]], idcs
            else:
                idx = scores.argmax(dim=-1)
                return self.voc.itos[idx], idx

    
    def target_idx(self, target):
        if isinstance(target, str):
            return torch.tensor(self.voc.stoi[target])
        
        else:
            return torch.tensor([self.voc.stoi[t] for t in target])

    def precision_sak_ap_rr(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, target: torch.Tensor, target_txt: str, k=10):
        """Computes: precision, success@k, rank, reciprocal rank, closest word to prediction, closest word to target"""
        # if (isinstance(a, torch.Tensor) and a.dim() > 1) or (a[0].dim() > 1): # batch mode
        #     return self.b_precision_sak_ap_rr(a, b, c, target, target_txt, k=k)

        pred_w, idcs_pred = self.closest_clf(a, b, c, ranks=True)
        idx_tgt = self.target_idx(target_txt).to(idcs_pred.device)
        precision = 1. if idcs_pred[0] == idx_tgt else 0.
        if isinstance(k, list) or isinstance(k, tuple):
            sak = [
                1. if idcs_pred[:k_] in idx_tgt else 0.
                for k_ in k
            ]
        else:
            sak = 1. if idcs_pred[:k] in idx_tgt else 0.
        r = (idx_tgt == idcs_pred).nonzero(as_tuple=True)[0].item() + 1
        rr = 1/r

        return precision, sak, r, rr, pred_w, target_txt
    
    def b_precision_sak_ap_rr(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, targets: torch.Tensor, target_txt: str, k=10):
        """Computes: precision, success@k, rank, reciprocal rank, closest word to prediction, closest word to target"""
        pred_w, idcs_pred = self.closest_clf(a, b, c, ranks=True)
        idx_tgt = self.target_idx(target_txt).to(idcs_pred.device)
        precision = (idx_tgt == idcs_pred[:,0]).float()#.mean()

        if isinstance(k, list) or isinstance(k, tuple):
            sak = [
                (idcs_pred[:,:k_] == idx_tgt.unsqueeze(-1)).any(dim=-1).float()
                for k_ in k
            ]
        else:
            sak = (idcs_pred[:,:k] == idx_tgt.unsqueeze(-1)).any(dim=-1).float()#.mean()
        r = (idx_tgt.unsqueeze(-1) == idcs_pred).nonzero(as_tuple=True)[1] + 1
        rr = 1/r

        return precision, sak, r, rr, pred_w, target_txt

def load_emb(args, target_module: nn.Module):
    if not args.transfer:
        return
    if args.transfer == "auto":
        model_path = get_clf_model_path(args.dataset, args.language, args.model_seed_id)
    else:
        model_path = args.transfer
    state_dict = load_cnn_emb(model_path)
    target_module.load_state_dict(state_dict)
def load_annc_(args, target_module: nn.Module):
    if not args.transfer:
        return
    if args.transfer == "auto":
        model_path = get_clf_model_path(args.dataset, args.language, args.model_seed_id)
    else:
        model_path = args.transfer
    state_dict = load_annc(model_path)
    target_module.load_state_dict(state_dict)

def main(args):
    # Get all the relevant file paths
    report_path = get_ret_annc_report_path(args.dataset, args.language, args.model_seed_id)
    fails_path = get_ret_annc_fails_path(args.dataset, args.language, args.model_seed_id)
    expe_tag = get_ret_annc_expe_tag(args.dataset, args.language, args.model_seed_id)
    model_seed = RANDOM_SEEDS[args.model_seed_id]
    if args.skip and os.path.exists(report_path):
        print(f"Report {report_path} exists, aborting")
        return
    
    fix_gpu_args(args)

    if hasattr(args, 'deterministic') and args.deterministic:
        logger.warning("Determinism (--deterministic True) does not guarantee reproducible results when changing the number of processes.")

    train_loader, val_loader, test_loader, encoder = prepare_data(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=RANDOM_SEEDS[args.split_seed_id])

    # if args.max_epochs is not None:
    #     args.max_epochs = args.max_epochs * ceil(args.nb_analogies_train / len(train_loader.dataset))

    # --- Define models ---
    seed_everything(model_seed, workers=True)
    char_emb_size = 64
    nn = RetANNcLightning(
        char_emb_size=char_emb_size,
        encoder=encoder,
        filters=args.filters,
        report_path=report_path,
        fails_path=fails_path)
    load_emb(args, nn.emb)
    load_annc_(args, nn.clf)


    with torch.no_grad():
        with embeddings_voc(nn.emb, train_loader.dataset.dataset, test_loader.dataset.dataset) as voc:
            trainer = pl.Trainer.from_argparse_args(args,
                logger = None,
                **get_trainer_kwargs(args)#, find_unused_parameters=True)
            )
            nn.voc = voc
            nn.voc.vectors = nn.voc.vectors.to(nn.device)
            nn.extra_info = {
                "timestamp": datetime.now().timestamp(),
                "seed": model_seed,
                "seed_id": args.model_seed_id,
                "lang": args.language,
                "dataset": args.dataset}
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
    if parser is None:
        parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--filters', '-f', type=int, default=128, help='The number of filters of the retrival model.')
    model_parser.add_argument('--model_seed_id', '-Vm', type=int, default=0, help='The model seed.')
    model_parser.add_argument('--split_seed_id', '-Vs', type=int, default=0, help='The data splitting seed.')

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch-size', '-b', type=int, default=512, help='Batch size.')
    dataset_parser.add_argument('--transfer', '-T', type=str, default="auto", help='The path of the model to load ("auto" to select the ANNc model corresponding to the current language/random seed).')
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
