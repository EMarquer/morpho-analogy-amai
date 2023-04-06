import logging
from math import ceil
from typing import Literal

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
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import CharErrorRate

##################################

from genmorpho import load_pytorch
from genmorpho.morpho_gen import AutoEncoder
from siganalogies import CharEncoder, enrich, SIG2016_LANGUAGES, SIG2019_HIGH
from annr import ANNr, AnalogyRegressionLoss
from utils import prepare_data, precision_sak_ap_rr, to_csv, append_csv, embeddings_voc
from utils.lightning import fix_gpu_args, get_trainer_kwargs

import os
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:128"
seed_everything(42, workers=True)

torch.cuda.memory.set_per_process_memory_fraction(0.9)

RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
VERSION = "debug"

class RetLightning(pl.LightningModule):
    def __init__(self, emb_model: AutoEncoder, char_encoder: CharEncoder,
            criterion: Literal["mse", "cosine embedding loss", "relative shuffle", "relative all", "all"]="relative shuffle"):
        super().__init__()
        self.save_hyperparameters(ignore=['emb_model'])
        self.char_encoder: CharEncoder = char_encoder
        self.ae: AutoEncoder = emb_model
        self.emb = self.ae.encoder
        word_emb_size = self.emb.output_size
        self.reg = ANNr(emb_size=word_emb_size, mode="ab!=ac")
        #self.reg = AnalogyRegression(emb_size=word_emb_size, mode="ab=ac")

        # # disable grad for autoencoder
        # for param in self.ae.parameters():
        #     param.requires_grad = False
        
        self.criterion = AnalogyRegressionLoss(criterion)
        self.voc = None

        self.save_folder = ""
        self.common_save_file = ""
        self.extra_info = dict()
        self.test_cer = CharErrorRate()

    def configure_optimizers(self):
        # @lightning method
        optimizer = torch.optim.Adam([{"params": self.reg.parameters(), "lr": 1e-3}])
        optimizer = torch.optim.Adam([{"params": self.reg.parameters(), "lr": 1e-3}, {"params": self.ae.parameters(), "lr": 1e-4}])
        optimizer = torch.optim.Adam([{"params": self.parameters(), "lr": 1e-3}])
        return optimizer

    def training_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a_emb = self.emb(a)
        b_emb = self.emb(b)
        c_emb = self.emb(c)
        d_emb = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        generation_loss = torch.tensor(0, device=a.device, dtype=float)

        epoch_factor = min(max(1-((self.trainer.current_epoch-1)/5), 0.01), 0.99)

        # positive examples
        for (a_, a_emb_), (b_, b_emb_), (c_, c_emb_), (d_, d_emb_) in enrich((a, a_emb), (b, b_emb), (c, c_emb), (d, d_emb)):
            d_emb_pred = self.reg(a_emb_, b_emb_, c_emb_)

            loss += self.criterion(a_emb_, b_emb_, c_emb_, d_emb_, d_emb_pred)
            # add generation loss
            pred_chars = self.ae.decoder(d_emb_pred, d_[:,:-1], apply_softmax=False)
            generation_loss +=  nn.functional.cross_entropy(pred_chars.transpose(-1,-2), d_[:,1:], ignore_index=self.ae.padding_index)

        loss = (loss * epoch_factor) + ((1-epoch_factor) * generation_loss)

        # actual interesting metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a_emb = self.emb(a)
        b_emb = self.emb(b)
        c_emb = self.emb(c)
        d_emb = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        
        # positive examples
        for (a_, a_emb_), (b_, b_emb_), (c_, c_emb_), (d_, d_emb_) in enrich((a, a_emb), (b, b_emb), (c, c_emb), (d, d_emb)):
            d_emb_pred = self.reg(a_emb_, b_emb_, c_emb_)

            #loss += self.criterion(a_emb_, b_emb_, c_emb_, d_emb_, d_emb_pred)
            # add generation loss
            pred_chars = self.ae.decoder(d_emb_pred, d_[:,:-1], apply_softmax=False)
            generation_loss =  nn.functional.cross_entropy(pred_chars.transpose(-1,-2), d_[:,1:], ignore_index=self.ae.padding_index)
            loss += generation_loss
        
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
        a_string = self.char_encoder.decode(a, pad_char='')
        b_string = self.char_encoder.decode(b, pad_char='')
        c_string = self.char_encoder.decode(c, pad_char='')
        d_string = self.char_encoder.decode(d, pad_char='')

        loss = torch.tensor(0, device=a.device, dtype=float)
        scores = []
        fails = []
        
        def log_fails(permutation, strings, d_e_, d_e_pred):
            a, b, c, d = permutation
            a_string, b_string, c_string, d_string = strings
            tgt_decode = self.ae.decoder.generate(d_e_,
                initial_character=self.char_encoder.BOS_ID,
                stop_character=self.char_encoder.EOS_ID,
                pad_character=self.char_encoder.PAD_ID,
                max_size=max(64, d.size(-1)),
                sample=True)
            tgt_string = self.char_encoder.decode(tgt_decode, pad_char='') # slow
            
            d_pred = self.ae.decoder.generate(d_e_pred,
                initial_character=self.char_encoder.BOS_ID,
                stop_character=self.char_encoder.EOS_ID,
                pad_character=self.char_encoder.PAD_ID,
                max_size=max(64, d.size(-1)),
                sample=True) # using generation
            pred_string = self.char_encoder.decode(d_pred, pad_char='')  # slow
            acc = torch.tensor([pred_string[i] == d_string[i][1:] for i in range(a.size(0))])
            self.test_cer.update(pred_string, [d_string[i][1:] for i in range(a.size(0))])

            score_dict = {
                'accuracy': acc,
            }
            if self.voc is not None:
                p, sak, r, rr, pred_w, tgt_w = precision_sak_ap_rr(d_e_pred, d_e_, self.voc, k=[3,5,10])
                score_dict = {**score_dict,
                    'precision': p,
                    'success@3': sak[0],
                    'success@5': sak[1],
                    'success@10': sak[2],
                    'rr': rr
                }

            mask = acc < 1
            indices = torch.arange(a.size(0), device=acc.device)[mask]
            for i in indices:
                fails.append({
                        "A": a_string[i],
                        "B": b_string[i],
                        "C": c_string[i],
                        "actual D": d_string[i],
                        "predicted D": pred_string[i],
                        "target D": tgt_string[i], # what D would look like if ANNr was perfect (to see what part of the failure is due to the decoder)
                    })

            scores.append(score_dict)

        # positive example, target is 1
        for (a_, a_string_, a_e_), (b_, b_string_, b_e_), (c_, c_string_, c_e_), (d_, d_string_, d_e_) in enrich((a, a_string, a_e), (b, b_string, b_e), (c, c_string, c_e), (d, d_string, d_e)):
            d_e_pred = self.reg(a_e_, b_e_, c_e_)
            loss += self.criterion(a_e_, b_e_, c_e_, d_e_, d_e_pred)
            log_fails((a_, b_, c_, d_), (a_string_, b_string_, c_string_, d_string_), d_e_, d_e_pred)

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

            row = {
                "cer": self.test_cer.compute().item(),
                "accuracy": torch.mean(torch.cat([score_dict["accuracy"] for score_dict in scores], dim=-1).to(float)).item(),
                **self.extra_info}
            if "precision" in scores[0].keys():
                m_precision = torch.mean(torch.cat([score_dict["precision"] for score_dict in scores], dim=-1))
                m_sak3 = torch.mean(torch.cat([score_dict["success@3"] for score_dict in scores], dim=-1))
                m_sak5 = torch.mean(torch.cat([score_dict["success@5"] for score_dict in scores], dim=-1))
                m_sak10 = torch.mean(torch.cat([score_dict["success@10"] for score_dict in scores], dim=-1))
                m_rr = torch.mean(torch.cat([score_dict["rr"] for score_dict in scores], dim=-1))

                row = {**row, "precision": m_precision.item(), "success@3": m_sak3.item(), "success@5": m_sak5.item(), "success@10": m_sak10.item(), "mrr": m_rr.item()}
            print(row)
            append_csv(self.common_save_file, row)
            #print(fails)
            print(os.path.join(self.save_folder, "fails.csv"))
            to_csv(os.path.join(self.save_folder, "summary.csv"), row)
            to_csv(os.path.join(self.save_folder, "fails.csv"), fails)
            #self.log("my_reduced_metric", mean, rank_zero_only=True)

def main(args):
    # the names defined here correspond to the files and folder of the results
    expe_group = f"ae_annr/{args.dataset}/{args.language}"
    common_summary_file = f"logs/{expe_group}/summary.csv"
    expe_name = f"{expe_group}/model{args.model_seed_id}-data{args.split_seed_id}"
    summary_folder = f"logs/{expe_name}"
    summary_file = f"{summary_folder}/summary.csv"
    if args.skip and os.path.exists(summary_file):
        print(f"{summary_file} exists, skip")
        return

    # === load genmorpho model ===
    ae, char_encoder = load_pytorch(args.dataset, args.language, model_seed_id=args.model_seed_id, data_seed_id=args.split_seed_id)

    # === load data ===
    seed_everything(RANDOM_SEEDS[args.split_seed_id], workers=True)
    train_loader, val_loader, test_loader, encoder = prepare_data(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=RANDOM_SEEDS[args.split_seed_id], pad_id=char_encoder.PAD_ID)
    # do some fixes to the arguments
    if args.max_epochs is not None:
        args.max_epochs = args.max_epochs * ceil(args.nb_analogies_train / len(train_loader.dataset))

    # === create ANNr ===
    seed_everything(RANDOM_SEEDS[args.model_seed_id], workers=True)
    annr = RetLightning(ae, char_encoder)#, criterion="mse")

    # === train ANNr ===
    tb_logger = pl.loggers.TensorBoardLogger('logs/', expe_name, version=VERSION)
    checkpoint_callback=ModelCheckpoint(
        filename=f"ret-{args.dataset}-{args.language}-b{args.batch_size}-{{epoch:02d}}",
        monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            checkpoint_callback,
        ],
        logger = tb_logger,
        **get_trainer_kwargs(args)#, find_unused_parameters=True)
    )
    seed_everything(RANDOM_SEEDS[args.model_seed_id], workers=True)
    trainer.fit(annr, train_loader, val_loader)

    # === test ANNr + generation ===
    import datetime
    with torch.no_grad():
        #with embeddings_voc(annr.emb, train_loader.dataset.dataset, test_loader.dataset.dataset) as annr.voc:
            annr.extra_info = {
                "best_model": checkpoint_callback.best_model_path,
                "model_seed" : RANDOM_SEEDS[args.model_seed_id],
                "split_seed" : RANDOM_SEEDS[args.split_seed_id],
                "POSIX_timestamp" : datetime.datetime.now().timestamp(),
                "language": args.language,
                "dataset": args.dataset,
                "epochs": trainer.current_epoch}
            annr.common_save_file = common_summary_file
            annr.save_folder = summary_folder
            trainer.test(annr, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
        #annr.voc = None


def add_argparse_args(parser):
    # argument parsing
    parser = pl.Trainer.add_argparse_args(parser)

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force_rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch_size', '-b', type=int, default=512, help='Batch size.')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already. Uses the summary.csv file in the model folder to determine if the model has been trained already on the language.', action='store_true')

    seed_parser = parser.add_argument_group("Autoencoder random seed arguments")
    seed_parser.add_argument('--model_seed_id', '-Vm', type=int, default=0, help='The model seed.')
    seed_parser.add_argument('--split_seed_id', '-Vs', type=int, default=0, help='The model seed.')
    
    return parser, dataset_parser, seed_parser

if __name__ == "__main__":
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser, dataset_parser, seed_parser = add_argparse_args(parser)

    args = parser.parse_args()

    # handle version details
    fix_gpu_args(args)

    # start the training script
    main(args)
