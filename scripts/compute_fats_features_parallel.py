import os
os.environ["MKL_NUM_THREADS"]="1"
print(os.environ["MKL_NUM_THREADS"])
import sys
sys.path.append("../libs")
from vvv_utils import parse_metadata, parse_light_curve_data, split_list_in_chunks, vvv_path
from features import compute_fats_features

df_meta = parse_metadata(experiment="ALL", merge_subclasses=True)
file_list = df_meta.index

from os.path import join
import pandas as pd
from joblib import Parallel, delayed, dump

result = Parallel(n_jobs=10)(delayed(compute_fats_features)(batch_names) for batch_names in split_list_in_chunks(file_list, 100))

with open(join(vvv_path, "features/features_fats_all.pkl"), "wb") as f:
    dump(pd.concat(result), f)
