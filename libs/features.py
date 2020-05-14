import numpy as np
import pandas as pd
from os.path import join, exists
from vvv_utils import parse_light_curve_data
import turbofats


def compute_fats_features(batch_names):
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
        lc_data = parse_light_curve_data(name)
        if lc_data is None:
            continue
        # Check that lc has more than 10 points
        if lc_data.shape[0] < 10:
            print("Light curve %s has less than 10 points, skipping" %(name))
            continue
        # Compute features                  
        feature_space = turbofats.NewFeatureSpace(feature_list=feature_list, 
                                                  data_column_names=["mag", "mjd", "err"])
        features.append(feature_space.calculate_features(lc_data))
    return pd.concat(features)