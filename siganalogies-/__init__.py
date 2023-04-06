from typing import Union

from .encoders import *

from .config import *
from .utils import enrich, generate_negative, random_sample_negative, n_pos_n_neg
from .utils_no_cp import (enrich as enrich_no_cp,
    generate_negative as generate_negative_no_cp,
    random_sample_negative as random_sample_negative_no_cp,
    n_pos_n_neg as n_pos_n_neg_no_cp)

from .sig_2016 import Sig2016Dataset as Dataset2016
from .sig_2016 import dataset_factory as dataset_factory_2016
from .sig_2019 import Sig2019Dataset as Dataset2019
from .sig_2019 import dataset_factory as dataset_factory_2019
from .multi_dataset import MultilingualDataset
from .multi_dataset import dataset_factory as dataset_factory_multi


def dataset_factory(dataset="2016", **kwargs) -> Union[Dataset2016, Dataset2019]:
    if str(dataset) == "2016":
        return dataset_factory_2016(**kwargs)
    else:
        return dataset_factory_2019(**kwargs)
