"""download https://nlp.stanford.edu/data/glove.6B.zip as `sem_baseline/.vector_cache/glove.6B.zip` for the Wikipedia dump"""

from functools import partial
import logging
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
from torchtext.vocab import GloVe, FastText, Vectors
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split, Subset

from os.path import exists, abspath, dirname, join

THIS_FOLDER = dirname(abspath(__file__))
CACHE_FOLDER = join(THIS_FOLDER, ".vector_cache")
IMPORTS_PATH = join(THIS_FOLDER, "..")

CLF_EXPE_TAG = "clf-sem/{dataset}/{language}/{seed_id}/{model}-{emb_size}{post_emb}{covered}"
DEFAULT_LOG_FOLDER = join(THIS_FOLDER, "..", "results")
DEFAULT_MODEL_FOLDER = join(THIS_FOLDER, "..", "models")

def get_clf_expe_tag(dataset=2019, language="english", seed_id=0, model="GloVe", emb_size=300, post_emb=False, covered=False):
    return CLF_EXPE_TAG.format(dataset=dataset, language=language, seed_id=seed_id, model=model, emb_size=emb_size, post_emb="-post_emb" if post_emb else "", covered="-covered" if covered else "")
def get_clf_model_path(dataset=2019, language="english", seed_id=0, model="GloVe", emb_size=300, post_emb=False, covered=False):
    return join(DEFAULT_MODEL_FOLDER, get_clf_expe_tag(dataset, language, seed_id, model=model, emb_size=emb_size, post_emb=post_emb, covered=covered), f"model.pkl")
def get_clf_report_path(dataset=2019, language="english", seed_id=0, model="GloVe", emb_size=300, post_emb=False, covered=False):
    return join(DEFAULT_MODEL_FOLDER, get_clf_expe_tag(dataset, language, seed_id, model=model, emb_size=emb_size, post_emb=post_emb, covered=covered), f"summary.csv")
def get_clf_fails_path(dataset=2019, language="english", seed_id=0, model="GloVe", emb_size=300, post_emb=False, covered=False):
    return join(DEFAULT_LOG_FOLDER, get_clf_expe_tag(dataset, language, seed_id, model=model, emb_size=emb_size, post_emb=post_emb, covered=covered), f"fails.csv")

try:
    from ..annc import ANNc
    from ..cnn_embeddings import CNNEmbedding
    from ..siganalogies import dataset_factory, enrich, generate_negative, n_pos_n_neg
    from ..utils import prepare_data, tpr_tnr_balacc_harmacc_f1
    from ..utils.logger import to_csv
    from ..utils.lightning import get_trainer_kwargs, fix_gpu_args
    from ..utils.ckpt import save_final_model
    from ..utils.data import collate, SPLIT_RANDOM_SEED
except ImportError:
    import sys
    sys.path.append(abspath(IMPORTS_PATH))
    from annc import ANNc
    from cnn_embeddings import CNNEmbedding
    from siganalogies import dataset_factory, enrich, generate_negative, n_pos_n_neg
    from utils import tpr_tnr_balacc_harmacc_f1
    from utils.logger import to_csv
    from utils.lightning import get_trainer_kwargs, fix_gpu_args
    from utils.ckpt import save_final_model
    from utils.data import collate, SPLIT_RANDOM_SEED

MODELS = {
    "GloVe6B50":  ("GloVe", 50),
    "GloVe6B100": ("GloVe", 100),
    "GloVe6B200": ("GloVe", 200),
    "GloVe6B300": ("GloVe", 300),
    "FastText":   ("FastText", 300),
    #"CNN+ANNc":   ("CNN+ANNc", 80),
}

COVERED_DATA_TRAIN_PATH = join(CACHE_FOLDER, "english-covered-train.pkl")
COVERED_DATA_DEV_PATH = join(CACHE_FOLDER, "english-covered-dev.pkl")
COVERED_DATA_TEST_PATH = join(CACHE_FOLDER, "english-covered-test.pkl")
COVERED_DATA_CHAR_TRAIN_PATH = join(CACHE_FOLDER, "english-covered-char-train.pkl")
COVERED_DATA_CHAR_DEV_PATH = join(CACHE_FOLDER, "english-covered-char-dev.pkl")
COVERED_DATA_CHAR_TEST_PATH = join(CACHE_FOLDER, "english-covered-char-test.pkl")
COVERED_VOCAB_PATH= join(THIS_FOLDER, "covered_vocab.txt")

def prepare_data(nb_analogies_train, nb_analogies_val, nb_analogies_test, batch_size = 32, force_rebuild=False, generator_seed=42, split_seed=SPLIT_RANDOM_SEED, pad_id=-1, word_encoder=None, filter_covered=False):
    '''Prepare the dataloaders for a given language.

    Arguments:
    nb_analogies_train -- The number of analogies to use (before augmentation) for the training.
    nb_analogies_val -- The number of analogies to use (before augmentation) for the validation (during the training).
    nb_analogies_test -- The number of analogies to use (before augmentation) for the testing (after the training).'''
    if filter_covered and word_encoder is None and exists(COVERED_DATA_TRAIN_PATH) and exists(COVERED_DATA_DEV_PATH) and exists(COVERED_DATA_TEST_PATH):
        train_data, val_data, test_data = torch.load(COVERED_DATA_TRAIN_PATH), torch.load(COVERED_DATA_DEV_PATH), torch.load(COVERED_DATA_TEST_PATH)
    elif filter_covered and word_encoder is not None and exists(COVERED_DATA_CHAR_TRAIN_PATH) and exists(COVERED_DATA_CHAR_DEV_PATH) and exists(COVERED_DATA_CHAR_TEST_PATH):
        train_data, val_data, test_data = torch.load(COVERED_DATA_CHAR_TRAIN_PATH), torch.load(COVERED_DATA_CHAR_DEV_PATH), torch.load(COVERED_DATA_CHAR_TEST_PATH)
    else:
        ## Train and test dataset
        dataset_ = dataset_factory(dataset="2019", language="english", mode="train-high", word_encoder=word_encoder, force_rebuild=force_rebuild)
        if filter_covered:
            # get the covered vocab
            if exists(COVERED_VOCAB_PATH):
                with open(COVERED_VOCAB_PATH, "r") as f:
                    covered_vocab = f.read().split("\n")
                logger.warning(f"Vocabulary coverage: {len(covered_vocab)/len(dataset_.word_voc):>6.2%}")
            else:
                glove = GloVe(name="6B", dim=50, cache=CACHE_FOLDER)
                ft=FastText(language="en", cache=CACHE_FOLDER)
                covered_vocab = [w for w in dataset_.word_voc if ((w.lower() in glove.stoi.keys()) or (w in glove.stoi.keys())) and ((w.lower() in ft.stoi.keys()) or (w in ft.stoi.keys()))]
                del glove
                del ft
                logger.warning(f"Vocabulary coverage: {len(covered_vocab)/len(dataset_.word_voc):>6.2%}")
                with open(COVERED_VOCAB_PATH, "w") as f:
                    f.write("\n".join(covered_vocab))

            # identify covered analogies (very slow for some reason)
            dataset_indices = [i for i, (a, b, c, d) in enumerate(dataset_) if ((a in covered_vocab) and (b in covered_vocab) and (c in covered_vocab) and (d in covered_vocab))]
            dataset = Subset(dataset_, dataset_indices)
            logger.warning(f"Analogy coverage: {len(dataset)/len(dataset_):>6.2%}")
        else:
            dataset=dataset_

        ## Data split
        lengths = [nb_analogies_train, nb_analogies_val, nb_analogies_test]
        if sum(lengths) > len(dataset):
            # 75% for training (70% for the actual training data, 5% for devlopment) and 25% for testing
            lengths = [int(len(dataset) * .70), int(len(dataset) * .05)]
            lengths.append(len(dataset) - sum(lengths)) # add the remaining data for testing
            lengths.append(0) # add a chunk with the remaining unused data
            logger.warning(f"Data is too small for the split {nb_analogies_train}|{nb_analogies_val}|{nb_analogies_test}, using {lengths[0]} (70%)|{lengths[1]} (5%)|{lengths[2]} (25%) instead.")
        else:
            lengths.append(len(dataset) - sum(lengths)) # add a chunk with the remaining unused data

        train_data, val_data, test_data, unused_data = random_split(dataset, lengths,
            generator=torch.Generator().manual_seed(split_seed))
        if word_encoder is None:
            torch.save(train_data, COVERED_DATA_TRAIN_PATH)
            torch.save(val_data,   COVERED_DATA_DEV_PATH)
            torch.save(test_data,  COVERED_DATA_TEST_PATH)
        else:
            torch.save(train_data, COVERED_DATA_CHAR_TRAIN_PATH)
            torch.save(val_data,   COVERED_DATA_CHAR_DEV_PATH)
            torch.save(test_data,  COVERED_DATA_CHAR_TEST_PATH)

    # Dataloader
    collate_fn = partial(collate, bos_id = dataset.word_encoder.BOS_ID, eos_id = dataset.word_encoder.EOS_ID, pad_id = pad_id) if word_encoder=="char" else (lambda x: list(zip(*x)))
    args = {
        "collate_fn": collate_fn,
        "num_workers": 4,
        "batch_size": batch_size,
        "persistent_workers": True
    }

    train_loader = DataLoader(train_data, generator=torch.Generator().manual_seed(generator_seed), shuffle=True, **args)
    val_loader = DataLoader(val_data, **args)#, generator=g_val)
    test_loader = DataLoader(test_data, **args)#, generator=g_test)

    return train_loader, val_loader, test_loader


import os
os.environ['PYTHONHASHSEED'] = str(42)
seed_everything(42, workers=True)

RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
VERSION = 1.0

DATASET = "2019"
LANGUAGE = "english"

def mask_valid(a, b, c, d):
    mask = [
        ((a_i == b_i) and (c_i == d_i)) or
        ((a_i == c_i) and (b_i == d_i))
        for a_i, b_i, c_i, d_i in zip(a, b, c, d)
    ]
    return mask




class ClfLightning(pl.LightningModule):
    def __init__(self, emb: Vectors, filters = 128, drop_fake_negative=True, report_path="", fails_path="", post_emb_size=0):
        super().__init__()
        self.save_hyperparameters()
        self.emb = emb

        if post_emb_size > 0:
            self.post_emb = nn.Linear(emb.vectors.size(-1), 80)
            self.clf = ANNc(emb_size=80, filters = filters)
        else:
            self.post_emb = None
            self.clf = ANNc(emb_size=emb.vectors.size(-1), filters = filters)
        
        self.criterion = nn.BCELoss()

        self.drop_fake_negative = drop_fake_negative

        self.report_path = report_path
        self.fails_path = fails_path
        self.extra_info = dict()
        self.test_performance = dict()

    def configure_optimizers(self):
        # @lightning method
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb.get_vecs_by_tokens(list(a), lower_case_backup=True).to(self.device)
        b = self.emb.get_vecs_by_tokens(list(b), lower_case_backup=True).to(self.device)
        c = self.emb.get_vecs_by_tokens(list(c), lower_case_backup=True).to(self.device)
        d = self.emb.get_vecs_by_tokens(list(d), lower_case_backup=True).to(self.device)

        if self.post_emb is not None:
            a = self.post_emb(a)
            b = self.post_emb(b)
            c = self.post_emb(c)
            d = self.post_emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        pos, neg = [], []

        pos_permutations, neg_permutations = n_pos_n_neg(a, b, c, d, filter_invalid=self.drop_fake_negative)

        # positive example, target is 1
        for a_, b_, c_, d_ in pos_permutations:
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            loss += self.criterion(is_analogy, expected)
            pos.append(is_analogy >= 0.5)

        # negative example, target is 0
        for a_, b_, c_, d_ in neg_permutations:
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
            loss += self.criterion(is_analogy, expected)
            neg.append(is_analogy < 0.5)

        pos = torch.cat(pos).view(-1).float()
        neg = torch.cat(neg).view(-1).float()

        # actual data to comute stats at the end
        tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()

        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)
        # actual interesting metrics
        self.log('train_loss',   loss,   on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))
        self.log('train_balacc', balacc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))
        self.log('train_f1',     f1,     on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb.get_vecs_by_tokens(list(a), lower_case_backup=True).to(self.device)
        b = self.emb.get_vecs_by_tokens(list(b), lower_case_backup=True).to(self.device)
        c = self.emb.get_vecs_by_tokens(list(c), lower_case_backup=True).to(self.device)
        d = self.emb.get_vecs_by_tokens(list(d), lower_case_backup=True).to(self.device)

        if self.post_emb is not None:
            a = self.post_emb(a)
            b = self.post_emb(b)
            c = self.post_emb(c)
            d = self.post_emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        pos, neg = [], []

        # positive example, target is 1
        for a_, b_, c_, d_ in enrich(a, b, c, d):
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            loss += self.criterion(is_analogy, expected)
            pos.append(is_analogy >= 0.5)

            # negative example, target is 0
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_):
                is_analogy = self.clf(a__, b__, c__, d__)

                expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
                loss += self.criterion(is_analogy, expected)
                neg.append(is_analogy < 0.5)

        pos = torch.cat(pos).view(-1).float()
        neg = torch.cat(neg).view(-1).float()

        # actual data to comute stats at the end
        tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()

        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)
        # actual interesting metrics
        self.log('val_loss',   loss,   on_step=False, on_epoch=True, prog_bar=True,  logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))
        self.log('val_balacc', balacc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))
        self.log('val_f1',     f1,     on_step=False, on_epoch=True, prog_bar=True,  logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch
        a_text,b_text,c_text,d_text = batch

        # compute the embeddings
        a = self.emb.get_vecs_by_tokens(list(a), lower_case_backup=True).to(self.device)
        b = self.emb.get_vecs_by_tokens(list(b), lower_case_backup=True).to(self.device)
        c = self.emb.get_vecs_by_tokens(list(c), lower_case_backup=True).to(self.device)
        d = self.emb.get_vecs_by_tokens(list(d), lower_case_backup=True).to(self.device)

        if self.post_emb is not None:
            a = self.post_emb(a)
            b = self.post_emb(b)
            c = self.post_emb(c)
            d = self.post_emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        pos, neg = [], []

        fails = []

        def log_fails(permutation_text, is_analogy, positive=True):
            a_text_,b_text_,c_text_,d_text_ = permutation_text
            fail_mask = is_analogy < 0.5 if positive else is_analogy >= 0.5

            for i, fail in enumerate(fail_mask):
                if fail:
                    fails.append({
                        "type": "False Negative" if positive else "False Positive",
                        "A": a_text_[i],
                        "B": b_text_[i],
                        "C": c_text_[i],
                        "D": d_text_[i],
                    })

        # positive example, target is 1
        for (a_, a_t_), (b_, b_t_), (c_, c_t_), (d_, d_t_) in enrich((a, a_text), (b, b_text), (c, c_text), (d, d_text)):
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones_like(is_analogy)
            loss += self.criterion(is_analogy, expected)
            pos.append(is_analogy >= 0.5)
            log_fails((a_t_, b_t_, c_t_, d_t_), is_analogy, positive=True)

            # negative example, target is 0
            for (a__, a_t__), (b__, b_t__), (c__, c_t__), (d__, d_t__) in generate_negative((a_, a_t_), (b_, b_t_), (c_, c_t_), (d_, d_t_)):
                is_analogy = self.clf(a__, b__, c__, d__)

                if self.drop_fake_negative:
                    m: torch.Tensor = mask_valid(a_t__, b_t__, c_t__, d_t__)
                    
                    is_analogy = is_analogy[[not m_ for m_ in m]]
                    expected = torch.zeros_like(is_analogy)
                    loss += self.criterion(is_analogy, expected)
                    neg.append(is_analogy < 0.5)
                    masked_tuple = tuple(zip(
                        *[
                        (a_t_i,b_t_i,c_t_i,d_t_i) for a_t_i,b_t_i,c_t_i,d_t_i,m_i in zip(a_t__, b_t__, c_t__, d_t__,m) if not m_i
                        ]
                    )) # (a_t__[~m], b_t__[~m], c_t__[~m], d_t__[~m])
                    log_fails(masked_tuple, is_analogy, positive=False)
                else:
                    expected = torch.zeros_like(is_analogy)
                    loss += self.criterion(is_analogy, expected)
                    neg.append(is_analogy < 0.5)
                    log_fails((a_t__, b_t__, c_t__, d_t__), is_analogy, positive=False)

        pos = torch.cat(pos).view(-1).float()
        neg = torch.cat(neg).view(-1).float()

        # actual data to comute stats at the end
        tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()
        self.log('true_positive',  tp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # true positive
        self.log('true_negative',  tn, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # true negative
        self.log('false_positive', fp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # false positive
        self.log('false_negative', fn, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # false negative
        
        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)
        # actual interesting metrics
        self.log('test_loss',   loss,   on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))
        self.log('test_balacc', balacc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))
        self.log('test_f1',     f1,     on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))

        # as averaging harmonc accuracies of the batches is not equivalent to the harmonic accuracy of the whole, the
        #    final averaged value of 'test_harmacc_approx' will only be an approximation of the harmonic accuracy.
        self.log('test_harmacc_approx', harmacc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=pos.size(0)+neg.size(0))


        return {
            "scores": {"tp": tp, "tn": tn, "fn": fn, "fp": fp},
            "fails": fails
        }

    def test_epoch_end(self, outputs):
        gathered = self.all_gather(outputs)

        # When logging only on rank 0, don't forget to add ``rank_zero_only=True`` to avoid deadlocks on synchronization.
        if self.trainer.is_global_zero:
            fails = []
            for gathered_ in gathered:
                fails.extend(gathered_["fails"])

            tp = sum(gathered_["scores"].get("tp", 0) for gathered_ in gathered).sum()
            tn = sum(gathered_["scores"].get("tn", 0) for gathered_ in gathered).sum()
            fp = sum(gathered_["scores"].get("fp", 0) for gathered_ in gathered).sum()
            fn = sum(gathered_["scores"].get("fn", 0) for gathered_ in gathered).sum()
            
            tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)

            self.test_performance = {"balacc": balacc.item(), "harmacc": harmacc.item(), "TPR": tpr.item(), "TNR": tnr.item(), "F1": f1.item(), **self.extra_info}
            print(self.test_performance)
            #append_pkl(self.common_save_path, row)
            to_csv(self.report_path, self.test_performance)
            to_csv(self.fails_path, fails)
            #self.log("my_reduced_metric", mean, rank_zero_only=True)

def generate_fasttext_file(name) -> Vectors:
    """Stores the embeddings of the encoder vocabulary and returns the vectors.
    
    Warning: If a word contains spaces, they are replaced by non-breaking spaces in the file.
        If a word contains non-breaking spaces, they will be interpreted as regular spaces when loading with `embeddings_voc`.
    """
    
    ft = FastText(language="en", cache=CACHE_FOLDER)

    with open(COVERED_VOCAB_PATH, "r") as f:
        covered_vocab = f.read().split("\n")

    with open(join(CACHE_FOLDER, name), 'w') as f:
        for word in covered_vocab:
            embedding = torch.squeeze(ft.get_vecs_by_tokens([word])).tolist()
            embedding = ' '.join(str(i) for i in embedding)
            f.write(f"{word} {embedding}\n")

def load_emb(model_variant, model_dim, distributed_barier=True):
    if "GloVe".lower() in model_variant.lower():
        return GloVe(name="6B", dim=model_dim, cache=CACHE_FOLDER)
    else:
        name="ft-filtered-en"
        try:
            ft_clamped = Vectors(name=name, cache=CACHE_FOLDER)
        except RuntimeError:
            from pytorch_lightning.utilities import rank_zero_only
            try:
                @rank_zero_only
                def _():
                    generate_fasttext_file(name)
                _()
                if distributed_barier and torch.distributed.is_initialized(): torch.distributed.barrier(group=torch.distributed.group.WORLD) # just to have a barrier, such that non rank 0 processes have access to the vectors file
            finally:
                ft_clamped = Vectors(name=name, cache=CACHE_FOLDER)
        return ft_clamped #FastText(language="en", cache=CACHE_FOLDER)

def main(args):
    model_variant, model_dim = MODELS[args.model_variant]

    # Get all the relevant file paths
    report_path = get_clf_report_path(DATASET, LANGUAGE, args.model_seed_id, model_variant, model_dim, post_emb=args.post_emb, covered=args.covered)
    fails_path =  get_clf_fails_path (DATASET, LANGUAGE, args.model_seed_id, model_variant, model_dim, post_emb=args.post_emb, covered=args.covered)
    model_path =  get_clf_model_path (DATASET, LANGUAGE, args.model_seed_id, model_variant, model_dim, post_emb=args.post_emb, covered=args.covered)
    expe_tag =    get_clf_expe_tag   (DATASET, LANGUAGE, args.model_seed_id, model_variant, model_dim, post_emb=args.post_emb, covered=args.covered)
    model_seed = RANDOM_SEEDS[args.model_seed_id]
    if args.skip and os.path.exists(report_path):
        print(f"Report {report_path} exists, aborting")
        return

    if hasattr(args, 'deterministic') and args.deterministic:
        logger.warning("Determinism (--deterministic True) does not guarantee reproducible results when changing the number of processes.")

    train_loader, val_loader, test_loader = prepare_data(
        args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=RANDOM_SEEDS[args.split_seed_id], word_encoder=None if model_variant!="CNN+ANNc" else "char", filter_covered=args.covered)

    # --- Define models ---
    seed_everything(model_seed, workers=True)
    char_emb_size = 64
    nn = ClfLightning(
        emb=load_emb(model_variant, model_dim) if model_variant!="CNN+ANNc" else CNNEmbedding(voc_size=len(train_loader.dataset.dataset.dataset.word_encoder.id_to_char), char_emb_size=char_emb_size),
        filters=args.filters,
        drop_fake_negative=not args.include_fake_negative,
        report_path=report_path,
        fails_path=fails_path,
        post_emb_size=80 if args.post_emb else 0)

    # --- Train model ---
    tb_logger = pl.loggers.TensorBoardLogger(DEFAULT_LOG_FOLDER, expe_tag, version=VERSION)
    checkpoint_callback=ModelCheckpoint(
        filename=f"pl-model-b{args.batch_size}-{{epoch:02d}}",
        monitor="val_balacc", mode="max", save_top_k=1)
    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[
            checkpoint_callback,
        ],
        logger = tb_logger,
        #auto_scale_batch_size = "binsearch",
        **get_trainer_kwargs(args)#, find_unused_parameters=True)
    )

    seed_everything(model_seed, workers=True)
    #trainer.tune(nn, train_loader, val_loader)
    trainer.fit(nn, train_loader, val_loader)

    #logger.info(f"best model path: {checkpoint_callback.best_model_path} (validation balanced accuracy: {checkpoint_callback.best_model_score:.3f})")
    
    with torch.no_grad():
        nn.extra_info = {
            "timestamp": datetime.now().timestamp(),
            "seed": model_seed,
            "seed_id": args.model_seed_id,
            "version": VERSION,
            "lang": LANGUAGE,
            "dataset": DATASET,
            "model_variant": model_variant,
            "emb_dim": model_dim,
            "covered_analogies_only": args.covered,
            #"variant": {"undef": "central permutation undefined", "bad": "central permutation bad" , "": "central permutation"}[args.cp]
        }
        #nn.common_save_path = "results/ret.pkl"
        trainer.test(nn, dataloaders=test_loader, ckpt_path="best")

    # load best model and save it at the right place
    state_dict = torch.load(checkpoint_callback.best_model_path, map_location="cpu")["state_dict"]
    state_dict_emb = args.model_variant #{k[len("emb."):]: v for k, v in state_dict.items() if k.startswith("emb.")}
    state_dict_clf = {k[len("clf."):]: v for k, v in state_dict.items() if k.startswith("clf.")}
    save_final_model(model_path, performance_dict=nn.test_performance, other_info=nn.extra_info, cnn_emb_state_dict=state_dict_emb, annc_state_dict=state_dict_clf)

def add_argparse_args(parser=None, return_all=False):
    if parser is None:
        parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--filters', '-f', type=int, default=128, help='The number of filters of the classification model.')
    model_parser.add_argument('--model-seed-id', '-Vm', type=int, default=0, help='The model seed.')
    model_parser.add_argument('--split-seed-id', '-Vs', type=int, default=0, help='The data splitting seed.')
    model_parser.add_argument('--model-variant', '-m', type=str, default="FastText", help='The embedding model to use.', choices=list(MODELS.keys()))
    model_parser.add_argument('--post-emb',  action='store_true', help='Use a linear layer to map the pre-trained embedding to an embedding of the same size as the usual CNN embedding model.')

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch-size', '-b', type=int, default=256, help='Batch size.')
    dataset_parser.add_argument('--include-fake-negative', help='By default, ignore negative permutations that end up being a:a::b:b or a:b::a:b. Add this flag to include said permutations.', action='store_true')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already.', action='store_true')
    dataset_parser.add_argument('--covered',  action='store_true', help='Use only covered analogies.')

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
