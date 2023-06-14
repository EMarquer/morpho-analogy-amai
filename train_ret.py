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
    from .utils import prepare_data, precision_sak_ap_rr, embeddings_voc
    from .utils.logger import to_csv
    from .utils.lightning import get_trainer_kwargs, fix_gpu_args
    from .utils.ckpt import get_clf_model_path, get_ret_model_path, get_ret_report_path, get_ret_expe_tag, get_ret_fails_path, DEFAULT_LOG_FOLDER, save_final_model, load_cnn_emb
except ImportError:
    from annr import ANNr, AnalogyRegressionLoss
    from annc import ANNc
    from cnn_embeddings import CNNEmbedding
    from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich
    from utils import prepare_data, precision_sak_ap_rr, embeddings_voc
    from utils.logger import to_csv
    from utils.lightning import get_trainer_kwargs, fix_gpu_args
    from utils.ckpt import get_clf_model_path, get_ret_model_path, get_ret_report_path, get_ret_expe_tag, get_ret_fails_path, DEFAULT_LOG_FOLDER, save_final_model, load_cnn_emb

import os
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:128"
seed_everything(42, workers=True)

torch.cuda.memory.set_per_process_memory_fraction(0.9)

RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
VERSION = 1.0

class RetLightning(pl.LightningModule):
    def __init__(self, char_emb_size, encoder, filters = 128, criterion: str="cosine embedding loss", freeze_emb=False, report_path="", fails_path=""):
        super().__init__()
        self.save_hyperparameters()
        self.emb = CNNEmbedding(voc_size=len(encoder.id_to_char), char_emb_size=char_emb_size)
        self.reg = ANNr(emb_size=self.emb.get_emb_size(), filters = filters, mode="ab!=ac")
        self.encoder = encoder
        
        self.criterion = criterion
        self.criterion_ = AnalogyRegressionLoss(variant=self.criterion)
        self.voc = None

        self.freeze_emb = freeze_emb
        
        self.report_path = report_path
        self.fails_path = fails_path
        self.extra_info = dict()
        self.test_performance = dict()
        self.force_eval_on_cpu = False

    def configure_optimizers(self):
        # @lightning method
        if self.freeze_emb:
            optimizer = torch.optim.Adam([
                {"params": self.reg.parameters(), "lr": 1e-3}])
        else:
            optimizer = torch.optim.Adam([
                {"params": self.emb.parameters(), "lr": 1e-5},
                {"params": self.reg.parameters(), "lr": 1e-3}])
        return optimizer

    def training_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)

        # positive examples
        for a_, b_, c_, d_ in enrich(a, b, c, d):
            d_pred = self.reg(a_, b_, c_)

            loss += self.criterion_(a_, b_, c_, d_, d_pred)

        # actual interesting metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        
        # positive examples
        for a_, b_, c_, d_ in enrich(a, b, c, d):
            d_pred = self.reg(a_, b_, c_)

            loss += self.criterion_(a_, b_, c_, d_, d_pred)
        
        # actual interesting metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a_e = self.emb(a)
        b_e = self.emb(b)
        c_e = self.emb(c)
        d_e = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        scores = []
        fails = []
        
        def log_fails(permutation, d_e_, d_pred):
            a, b, c, d = permutation
            
            #with elapsed_timer() as t:
            p, sak, r, rr, pred_w, tgt_w = precision_sak_ap_rr(d_pred, d_e_, self.voc, k=[3,5,10])

            mask = p < 1
            indices = torch.arange(a.size(0), device=p.device)[mask]
            for i in indices:
                fails.append({
                        "A": self.encoder.decode(a[i], pad_char=''),
                        "B": self.encoder.decode(b[i], pad_char=''),
                        "C": self.encoder.decode(c[i], pad_char=''),
                        "actual D": self.encoder.decode(d[i], pad_char=''),
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

        # positive example, target is 1
        for (a_, a_e_), (b_, b_e_), (c_, c_e_), (d_, d_e_) in enrich((a, a_e), (b, b_e), (c, c_e), (d, d_e)):
            d_pred = self.reg(a_e_, b_e_, c_e_)
            loss += self.criterion_(a_e_, b_e_, c_e_, d_e_, d_pred)
            if self.force_eval_on_cpu:
                d_e_, d_pred = d_e_.cpu(), d_pred.cpu()
            log_fails((a_, b_, c_, d_), d_e_, d_pred)

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
    # Get all the relevant file paths
    report_path = get_ret_report_path(args.dataset, args.language, args.model_seed_id)
    fails_path = get_ret_fails_path(args.dataset, args.language, args.model_seed_id)
    model_path = get_ret_model_path(args.dataset, args.language, args.model_seed_id)
    expe_tag = get_ret_expe_tag(args.dataset, args.language, args.model_seed_id)
    model_seed = RANDOM_SEEDS[args.model_seed_id]
    if args.skip and os.path.exists(report_path):
        print(f"Report {report_path} exists, aborting")
        return

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
    nn = RetLightning(
        char_emb_size=char_emb_size,
        encoder=encoder,
        filters=args.filters,
        criterion=args.criterion,
        report_path=report_path,
        fails_path=fails_path)
    load_emb(args, nn.emb)

    # --- Train model ---
    tb_logger = pl.loggers.TensorBoardLogger(DEFAULT_LOG_FOLDER, expe_tag, version=VERSION)
    checkpoint_callback=ModelCheckpoint(
        filename=f"pl-model-b{args.batch_size}-{{epoch:02d}}",
        monitor="val_loss", mode="min", save_top_k=1)
    
    # with emb frozen
    if args.freeze_emb:
        nn.freeze_emb = True
        trainer = pl.Trainer.from_argparse_args(args,
            callbacks=[
                EarlyStopping(monitor="val_loss"),
                checkpoint_callback,
            ],
            logger = tb_logger,
            **get_trainer_kwargs(args)#, find_unused_parameters=True)
        )

        seed_everything(model_seed, workers=True)
        trainer.fit(nn, train_loader, val_loader)
        epochs_no_emb = nn.current_epoch + 1
    else:
        epochs_no_emb = 0

    # with emb unfrozen
    if args.max_epochs is not None:
        args.max_epochs = args.max_epochs - epochs_no_emb
    nn.freeze_emb = False

    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            checkpoint_callback,
        ],
        logger = tb_logger,
        **get_trainer_kwargs(args)#, find_unused_parameters=True)
    )

    seed_everything(model_seed, workers=True)
    trainer.fit(nn, train_loader, val_loader)

    with torch.no_grad():
        with embeddings_voc(nn.emb, train_loader.dataset.dataset, test_loader.dataset.dataset) as voc:
            trainer = pl.Trainer.from_argparse_args(args,
            logger = None,
            **get_trainer_kwargs(args)#, find_unused_parameters=True)
            )
            
            import time
            start_time = time.time()
            
            nn.voc = voc
            nn.voc.vectors = nn.voc.vectors.to(nn.device)
            nn.extra_info = {
                "timestamp": datetime.now().timestamp(),
                "seed": model_seed,
                "seed_id": args.model_seed_id,
                "lang": args.language,
                "dataset": args.dataset,
                "criterion": args.criterion,
                #"transfer": args.transfer,
                "epochs_emb_frozen": epochs_no_emb,
                "epochs_total": epochs_no_emb + nn.current_epoch + 1}
            try:
                trainer.test(nn, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
            except RuntimeError as e: # Automatically falback to cpu if vocabulary is too large
                if 'out of memory' not in str(e).lower():
                    raise RuntimeError(e)
                nn.force_eval_on_cpu = True
                trainer.test(nn, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
            nn.voc = None
            
            print("--- %s seconds ---" % (time.time() - start_time))

    # load best model and save it at the right place
    state_dict = torch.load(checkpoint_callback.best_model_path, map_location="cpu")["state_dict"]
    state_dict_emb = {k[len("emb."):]: v for k, v in state_dict.items() if k.startswith("emb.")}
    state_dict_reg = {k[len("reg."):]: v for k, v in state_dict.items() if k.startswith("reg.")}
    save_final_model(model_path, performance_dict=nn.test_performance, other_info=nn.extra_info, cnn_emb_state_dict=state_dict_emb, annr_state_dict=state_dict_reg)

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
    dataset_parser.add_argument('--no-freeze-emb', help='Freeze embedding until convergence, then unfreeze it.',  action='store_false', dest="freeze_emb")
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch-size', '-b', type=int, default=512, help='Batch size.')
    dataset_parser.add_argument('--transfer', '-T', type=str, default="auto", help='The path of the model to load ("auto" to select the ANNc model corresponding to the current language/random seed).')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already.', action='store_true')
    dataset_parser.add_argument('--criterion', '-C', type=str, default="relative shuffle", choices=["cosine embedding loss", "relative shuffle", "relative all", "all"], help='The training loss to use.')

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
