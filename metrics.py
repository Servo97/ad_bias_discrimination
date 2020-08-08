import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import pickle
from math import cos, pi

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

Ad_dict = {'Clothing & Shoes': "A01_",
            'Automative':"A02_",
            'Baby Products':"A03_",
            'Health & Beauty': "A04_",
            'Media (BMVD)':'A065_',
            'Consumer Electroncis':'A06_',
            'Console & Video Games':'A07_',
            'DIY & Tools':"A08_",
            "Garden & Outdoor living": "A09_",
            "Grocery":"A10_",
            "Kitchen & Home":"A11_",
            "Betting":"A12_",
            "Jewellery & Watches":"A13_",
            "Musical Instruments":"A14_",
            "Office Products":"A15_",
            "Pet Supplies":"A16_",
            "Computer Software":"A17_",
            "Sports & Outdoors":"A18_",
            "Toys & Games":"A19_",
            "Dating Sites":"A20_"
        }

def metrics_from_df(df:pd.DataFrame, confidence_threshold=0):
    """Drop examples with probability < confidence_threshold from calc"""
    y_true = df['y_pred']
    y_pred = df['y_test']
    cnf_matrix = confusion_matrix(y_true, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return {"TPR": TPR, "TNR": TNR, "PPV": PPV, "NPV": NPV, "FPR": FPR, "FNR": FNR, "FDR": FDR, "ACC": ACC}

class GroupFairnessMetrics:
    def __init__(self, model_inference_data:pd.DataFrame, protected_feature:str):
        self._df, self._pf, = model_inference_data, protected_feature
        self._base_metrics = "fairness_metrics_per_class"
        self._pf_metrics_df = self._df.groupby(self._pf).apply(metrics_from_df).to_frame(self._base_metrics)

    def fetch_base_metrics(self):
        return self._pf_metrics_df

    def equal_opportunity_difference(self, pg_lbl:str, upg_lbl:str, rating_class=1):
        r"""TPR{unprivileged} - TPR{privileged} ideally should be zero"""
        upg_opp = self._pf_metrics_df.loc[upg_lbl][self._base_metrics]["TPR"][rating_class]
        pg_opp = self._pf_metrics_df.loc[pg_lbl][self._base_metrics]["TPR"][rating_class]
        return upg_opp - pg_opp

    def average_odds_difference(self, pg_lbl:str, upg_lbl:str, rating_class=1):
        """Average of difference in FPR and TPR for unprivileged and privileged groups"""
        tpr_diff = self.equal_opportunity_difference(pg_lbl, upg_lbl, rating_class)
        upg_fpr = self._pf_metrics_df.loc[upg_lbl][self._base_metrics]["FPR"][rating_class]
        pg_fpr = self._pf_metrics_df.loc[pg_lbl][self._base_metrics]["FPR"][rating_class]
        fpr_diff = upg_fpr - pg_fpr
        return 0.5 * (fpr_diff + tpr_diff)
    
def plot_for_metric_class(metric_df:pd.DataFrame, metric:str="FPR", rating_class:int=1):
    """Generates plot for metric and given rating_class from metric_df indexed by dimension of interest"""
    plot_df = metric_df.apply(lambda m: m["fairness_metrics_per_class"][metric][rating_class], axis=1)
    plot_df = plot_df.reset_index().rename({0: metric}, axis=1)
    return plot_df

def get_metrics(result_metrics, AD_type, BiasCategory):
    if AD_type!="Overall":
        result_metrics = result_metrics[result_metrics['AdId'].str.startswith(Ad_dict[AD_type])]
    class_GFM = GroupFairnessMetrics(result_metrics, BiasCategory)
    class_metrics_df = result_metrics.groupby(BiasCategory).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
    plot_fpr = plot_for_metric_class(class_metrics_df)['FPR'].tolist()
    plot_fpr_labels = plot_for_metric_class(class_metrics_df)[BiasCategory].tolist()
    title_fpr = f"{BiasCategory} FPR"
    x_label = BiasCategory
    y_label = "FPR"
    if BiasCategory=="Gender":
        EOD = class_GFM.equal_opportunity_difference("F","M")
        AOD = class_GFM.average_odds_difference("F","M")
    elif BiasCategory=="Age":
        EOD = class_GFM.equal_opportunity_difference(0,1)
        AOD = class_GFM.average_odds_difference(0,1)
    elif BiasCategory=="Income":
        EOD = class_GFM.equal_opportunity_difference(0,2)
        AOD = class_GFM.average_odds_difference(0,2)
    return (plot_fpr, plot_fpr_labels, x_label, y_label, title_fpr), EOD, AOD