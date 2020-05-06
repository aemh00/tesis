import numpy as np
import pandas as pd
from os.path import join, exists
import turbofats


def read_light_curve(path, column_names_lc = ["mjd", "mag", "err"]):
    """
    Receives a path
    
    Returns a dataframe with the light curve data (if found)
    """
    if not exists(path):
        print("File not found: %s" %(path))
        return None
    # Import from dat file and remove rows with nan
    lc_data = pd.read_csv(path, header=None, delim_whitespace=True, 
                          comment='#', names=column_names_lc).dropna()
    lc_data.sort_values(by="mjd", inplace=True)
    return lc_data

def compute_features(batch_names, data_path):
    """
    Receives a list of file names and a path
    
    Returns a dataframe with the features
    """
    # TODO: STUDY BIASED FEATURES
    feature_list = ['CAR_sigma','CAR_mean', 'Meanvariance', 'Mean', 'PercentDifferenceFluxPercentile',
                         'PercentAmplitude', 'Skew', 'AndersonDarling', 'Std', 'Rcs', 'StetsonK',
                         'MedianAbsDev', 'Q31', 'Amplitude', 'PeriodLS_v2', 'Harmonics',
                'Autocor_length', 'SlottedA_length', 'StetsonK_AC',  'Con', 'Beyond1Std', 
                'SmallKurtosis', 'MaxSlope','MedianBRP', 'PairSlopeTrend', 'FluxPercentileRatioMid20',
                'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65',
                'FluxPercentileRatioMid80', 'LinearTrend', 'Eta_e', 'Period_fit_v2', 'PeriodPowerRate',
                'Psi_CS_v2', 'Psi_eta_v2', 'StructureFunction_index_21', 'Pvar', 'StructureFunction_index_31',
                'ExcessVar', 'IAR_phi']
    features = []
    for name in batch_names:
        # Check that name is valid
        if name is None:
            continue
        # Read light curve
        path_to_file = join(data_path, name+'.dat')
        lc_data = read_light_curve(path_to_file)
        if lc_data is None:
            continue
        # Check that lc has more than 10 points
        if len(lc_data.index.unique()) < 10:
            print("Light curve %s has less than 10 points, skipping" %(name))
            continue
        # Create index column for FATS
        lc_data.index = [name]*len(lc_data.index.unique())
        # Compute features                  
        feature_space = turbofats.NewFeatureSpace(feature_list=feature_list, 
                                                  data_column_names=["mag", "mjd", "err"])
        features.append(feature_space.calculate_features(lc_data))
    return pd.concat(features)