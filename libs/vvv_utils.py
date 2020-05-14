import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join, exists
from os import listdir
from itertools import zip_longest 
from sklearn.model_selection import StratifiedShuffleSplit

vvv_path = "/home/shared/astro/VVV/"
vvv_header = "ID_VVV", "ID_ogle", "distancia espacial en grados", "<Ks>", "P1", "P2", "ID_ogle2", "P_ogle"

vvv_class_names = ['ell', 'ecl_nc', 'ecl_c', 'cepcl', 'cept2', 'rrab', 'rrc']
vvv_sub_class = {'ell': 'binary', 'ecl_nc' :'binary', 'ecl_c': 'binary', 
                 'cepcl': 'cepheid', 'cept2' : 'cepheid', 
                 'rrab': 'rrlyrae', 'rrc': 'rrlyrae'}

vvv_name2label = {vvv_class_names[i]: i for i in range(len(vvv_class_names))}
vvv_label2name = {i: vvv_class_names[i] for i in range(len(vvv_class_names))}

meta_dict_ceph = {'ceph_train/o4_ell_pmay1.fnl2': vvv_class_names[0],
                  'ceph_train/o4_ecl_nc_pmay1.fnl2': vvv_class_names[1], 
                  'ceph_train/o4_ecl_nc_p05.fnl2': vvv_class_names[1],
                  'ceph_train/o4_ecl_c_pmay1.fnl2': vvv_class_names[2],             
                  'ceph_train/bonafide_cepcl.fnl2': vvv_class_names[3],
                  'ceph_train/bonafide_cept2.fnl2': vvv_class_names[4],
                  'ceph_train/cep_dek19_t1_candidates.fnl2': vvv_class_names[3],
                  'ceph_train/cep_dek19_t2_candidates.fnl2': vvv_class_names[4]
                 }

meta_dict_rrl = {'rrl_train/o4_ell_pmen1.fnl2': vvv_class_names[0],
                 'rrl_train/o4_ecl_nc_pmen1_05.fnl2': vvv_class_names[1],
                 'rrl_train/o4_ecl_nc_pmen1.fnl2': vvv_class_names[1],
                 'rrl_train/o4_ecl_c_pmen1.fnl2': vvv_class_names[2],             
                 'rrl_train/o4_rrab.fnl2': vvv_class_names[5],
                 'rrl_train/o4_rrc.fnl2': vvv_class_names[6]
                }


def parse_metadata(experiment="ALL", merge_subclasses=True):
    """
    experiment: string
        CEPH: Metadata para clasificación de Cefeidas (Javier Minniti)
        RRL: Metadata para clasificación de RRL (Rodrigo Contreras)
        ALL: Merge de ambos training sets
    merge_subclasses: bool
        Separar en macro clases (Binarias, Cefeidas, RR Lyrae) o en sub-clases
    """
    if experiment == "CEPH":
        meta_dict = meta_dict_ceph
    elif experiment == "RRL":
        meta_dict = meta_dict_rrl
    elif experiment == "ALL":
        meta_dict = meta_dict_ceph.copy()
        meta_dict.update(meta_dict_rrl)
    else:
        raise ValueError("Options are CEPH, RRL or ALL")
        
    df_meta = []
    for  meta_file, string_label in meta_dict.items():
        df_meta_tmp = pd.read_csv(join(vvv_path, "metadata", meta_file), 
                                  names=vvv_header, delim_whitespace=True, comment='#')
        if not merge_subclasses:
            df_meta_tmp["label"] = string_label
        else:
            df_meta_tmp["label"] = vvv_sub_class[string_label]
        df_meta_tmp.set_index("ID_VVV", inplace=True)
        df_meta.append(df_meta_tmp[["P1", "P2", "P_ogle", "label"]])
    df_meta = pd.concat(df_meta)
    print("%d light curve metadata collected" %(len(df_meta)))
    return df_meta


def parse_light_curve_data(light_curve_id, column_names_lc = ["mjd", "mag", "err"], raise_error=True):
    """
    light_curve_id is a VVV id string, example "b221_201_22183"
    
    Returns a dataframe with the light curve data or None (if not found)
    """
    path_to_light_curve = join(vvv_path, "light_curves", light_curve_id+".dat")
    if not exists(path_to_light_curve):
        if raise_error:
            raise FileNotFoundError("File not found at: %s" %(path_to_light_curve))
        return None
    # Parse dat file and remove rows with nan
    lc_data = pd.read_csv(path_to_light_curve, header=None, delim_whitespace=True, 
                          comment='#', names=column_names_lc).dropna()
    lc_data.sort_values(by="mjd", inplace=True)
    # Set index as id (needed for FATS)
    lc_data.index = [light_curve_id]*len(lc_data.index.unique())
    return lc_data

def plot_light_curve(lc_dataframe, period=None, figsize=(5, 3)):
    """
    lc_dataframe is a pd.DataFrane with "mjd", "mag" and "err" columns
    
    If period is given, the light curve is folded before plotting
    """
    mjd, mag, err = lc_dataframe[["mjd", "mag", "err"]].values.T
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    if not period is None: # Plot folded light curve
        phi = np.mod(mjd, period)/period
        ax.errorbar(np.hstack((phi, phi+1)), np.hstack((mag, mag)),
                    np.hstack((err, err)), fmt='.')
        ax.set_xlabel("Phase")
    else:  # Plot unfolded light curve
        ax.errorbar(mjd, mag, err, fmt='.')
        ax.set_xlabel("Time [MJD]")
    ax.invert_yaxis()
    ax.set_ylabel("Magnitude")
    ax.set_title(lc_dataframe.index[0])    
                         

def get_train_test_ids(metadata_dataframe, random_seed=0):
    """    
    Returns tuple with train and test indexes (pd.Index, pd.Index)
    """
    vvv_ids = metadata_dataframe.index
    vvv_labels = np.array(metadata_dataframe["label"].values)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed)
    train_idx, test_idx = next(sss.split(vvv_ids, vvv_labels))
    return vvv_ids[train_idx], vvv_ids[test_idx]


def split_list_in_chunks(iterable, chunk_size, fillvalue=None):
    """
    Receives an iterable object (i.e. list) and a chunk size
    
    Returns an iterable object with the same elements on of the original but arranged in chunks 
    """
    args = [iter(iterable)] * chunk_size
    return zip_longest(*args, fillvalue=fillvalue)