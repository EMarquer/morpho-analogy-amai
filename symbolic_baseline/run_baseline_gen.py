import os, sys
from statistics import mean

from numpy import iterable
from levenshtein_distance import lp

# Change the current working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import logging
logging.getLogger("").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

import symbolic_baseline.murena.analogy as murena
import symbolic_baseline.alea.alea as alea
import pandas as pd
from multiprocessing import Pool

from utils.data import prepare_dataset
from utils import tpr_tnr_balacc_harmacc_f1
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich, generate_negative, n_pos_n_neg

os.environ['PYTHONHASHSEED'] = str(42)
MODEL_RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
RHO = 100
"""
source ~/miniconda3/bin/activate nn-morpho-analogy ; cd ~/orpailleur/emarquer/nn-morpho-analogy ; python run_baseline_gen.py arabic murena
"""

### multiprocessing wrappers ###
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired, concurrent
import multiprocessing
#multiprocessing.set_start_method("spawn")

from collections import Counter

def process_one_perm(a,b=None,c=None,d=None, model_name="murena"):
    if b is None: a,b,c,d = a
    if model_name == "murena":
        results = murena.solveAnalogy(a,b,c)
        results = [results[0][i][0].strip() for i in range(len(results[0]))][:1]

        results = list(set(results))

        return 1. if (d in results) else 0., max(lp(result, d) for result in results) if len(results) > 0 else 0
    
    elif model_name == "alea":
        results = alea.alea(a,b,c,rho=RHO)
        results = [result.strip() for result in results]
        results = [w for w, count in Counter(results).most_common(1)]
        
        results = list(set(results))

        return 1. if (d in results) else 0., max(lp(result, d) for result in results) if len(results) > 0 else 0

from siganalogies.abstract_analogy_dataset import AbstractAnalogyDataset
from siganalogies.encoders import NO_ENCODER
from functools import partial
from tqdm import tqdm # for progress bar

def run_multiprocessing(model_name, test_data, num_processes=multiprocessing.cpu_count(), timeout=5, error_as_failure=True, progress_bar=False):
    analogies = [_abcd for abcd in test_data for _abcd in enrich(*abcd)]
    _ps = []
    _lps = []
    with ProcessPool(max_workers=num_processes) as pool:
        future = pool.map(partial(process_one_perm, model_name=model_name), analogies, timeout=timeout)
        it = future.result()
        pool.close()
    
        fails = 0
        total = len(test_data) * 8
        if progress_bar: pbar = tqdm(total=total, desc="positive")
        while True:
            _p, _lp = None, None
            try:
                _p, _lp = next(it)
            except StopIteration:
                break
            except TimeoutError as error:
                #logger.error("Function took longer than %d seconds" % error.args[1])
                if error_as_failure: _p, _lp = 0., 0.
                fails += 1
            except ProcessExpired as error:
                #logger.error("%s. Exit code: %d" % (error, error.exitcode))
                if error_as_failure: _p, _lp = 0., 0.
                fails += 1
            except Exception as error:
                logger.error("Function raised %s" % error)
                logger.error(error.traceback)  # Python's traceback of remote process
                _p, _lp = 0., 0.
                fails += 1

            if _p is not None:
                _ps.append(_p) # precision
                _lps.append(_lp) # levenstein
            if progress_bar: pbar.update()
            if progress_bar: pbar.set_postfix({"failures": fails})
        if progress_bar: pbar.close()
        #pool.close()
    return _ps, _lps, fails, total

def precision_sak_ap_rr(r_from_0, k=10):
    """Computes: precision, success@k, rank, reciprocal rank, closest word to prediction, closest word to target"""
    precision = 1. if r_from_0==0 else 0.
    sak = 1. if r_from_0<k else 0.
    r = r_from_0 + 1
    rr = 1/r

    return precision, sak, r, rr

STORE_PREDS = False
def run_model(model_name, test_data, progress_bar=False):
    """All precision measures are tested at k=10 for positive samples and k=1 for negative samples"""

    logger.info("Starting processing of the test data:")
    _ps, _lps, fails, total = run_multiprocessing(model_name=model_name, test_data=test_data, num_processes=args.processes, timeout=args.timeout, progress_bar=progress_bar)
    logger.info("Done!")

    # generation results
    m_p, m_lp = mean(_ps), mean(_lps)
    return {
        "fails": fails,
        "total": total,

        "precision": m_p,
        "Lp": m_lp,
    }, (m_p, m_lp)


def output_file_name(args):
    return f"results/baselines-gen/{args.dataset}-{args.language}/{args.model}-{args.version}-{args.nb_analogies_test}.csv"

def main(args):
    logger.info(f"Processing baseline {args.model} on {args.language}...")
    train_data, val_data, test_data, dataset = prepare_dataset(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.force_rebuild, split_seed=MODEL_RANDOM_SEEDS[args.version])

    def dataset_to_no_char_but_no_rebuild(dataset: AbstractAnalogyDataset):
        dataset.word_encoder = NO_ENCODER
    dataset_to_no_char_but_no_rebuild(train_data.dataset)
    dataset_to_no_char_but_no_rebuild(val_data.dataset)
    dataset_to_no_char_but_no_rebuild(test_data.dataset)
    dataset_to_no_char_but_no_rebuild(dataset)

    x, y = test_data.dataset.analogies[test_data.indices[2]]
    print(test_data.dataset.raw_data[x])
    print(test_data.dataset.raw_data[y])
    print(test_data.dataset[test_data.indices[2]])

    record, (m_p, m_lp) = run_model(args.model, test_data, progress_bar=args.progress_bar)

    logger.info(f"Baseline {args.model} on {args.language}:\n{record}.")
    filename = output_file_name(args)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame.from_records([record]).to_csv(filename)
    logger.info(f"Processing baseline {args.model} on {args.language} done.")

if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--model', '-m', type=str, default="all", help='The baseline model to use.', choices=["all", "murena", "alea"])

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--recompute', help='Force the computation of the result even if already computed (combination version, nb-analogies-test, language, model) of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--version', '-V', type=int, default=0, help='The experiment version, also corresponding to the random seed to use.')
    dataset_parser.add_argument('--processes', '-p', type=int, default=-1, help='The maximum number of processes to use.')
    dataset_parser.add_argument('--progress-bar', help='Display a progress bar.', action='store_true')
    dataset_parser.add_argument('--timeout', '-o', type=int, default=10, help='The maximum time per processes.')

    args = parser.parse_args()

    if args.processes < 0: args.processes = multiprocessing.cpu_count()

    if args.model == "all":
        for model in ["murena", "alea"]:
            args.model = model
            if not os.path.exists(output_file_name(args)) or args.recompute:
                main(args)
    else:
        if not os.path.exists(output_file_name(args)) or args.recompute:
            main(args)
