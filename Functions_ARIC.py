import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_categorical_dtype
)
from joblib import Parallel, delayed
import math
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score, f1_score, accuracy_score,precision_recall_curve, auc,roc_curve, confusion_matrix,precision_score,recall_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imodels import RuleFitClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve, roc_curve, confusion_matrix, make_scorer)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imodels.rule_set.rule_fit import RuleFitClassifier
from statsmodels.stats.descriptivestats import sign_test


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.stats import spearmanr, binomtest, bootstrap
import xmlrpc.client
import time
import re
# from gams import GamsWorkspace
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


######################################
###### Preprocessing Functions ######
######################################

# List of required variables (the common columns we need)
required_vars = ['ETHANL03', 'CURSMK01', 'MOVE', 'P_CARB', 'P_PROT', 'P_SFAT', 'P_TFAT','BMI01', 'CHOL', 'DFIB', 'TOTCAL03', 'HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31', 'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09', 'CIGTYR01', 'ELEVEL01', 'GENDER','RACEGRP', 'V1AGE01', 'DIABTS', 'HYPERT04', 'HYPTMDCODE01', 'CHOLMDCODE01','ANTA01', 'CIGT01', 'HYPTMD01', 'ANTICOAGCODE01', 'ASPIRINCODE01', 'STATINCODE01'] #+Outcomes
changable_feature = ['ETHANL03', 'CURSMK01', 'MOVE', 'P_CARB', 'P_PROT', 'P_SFAT', 'P_TFAT','BMI01', 'CHOL', 'DFIB', 'TOTCAL03']
ind_changable = ['HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31', 'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09']
unchangeable= [ 'CIGTYR01', 'ELEVEL01', 'GENDER','RACEGRP', 'V1AGE01', 'DIABTS', 'HYPERT04', 'HYPTMDCODE01', 'CHOLMDCODE01','ANTA01', 'CIGT01', 'HYPTMD01', 'ANTICOAGCODE01', 'ASPIRINCODE01', 'STATINCODE01']
changable_feature_ordinal = ['ETHANL03_ordinal', 'CURSMK01_ordinal', 'MOVE_ordinal','P_CARB_ordinal', 'P_PROT_ordinal', 'P_SFAT_ordinal', 'P_TFAT_ordinal','BMI01_ordinal', 'CHOL_ordinal', 'DFIB_ordinal', 'TOTCAL03_ordinal']
ind_changable_ordinal = ['HDLSIU02_ordinal', 'LDLSIU02_ordinal', 'TCHSIU01_ordinal', 'TRGSIU01_ordinal','SBPA21_ordinal', 'SBPA22_ordinal', 'ANTA07A_ordinal','ANTA07B_ordinal', 'ECGMA31_ordinal', 'HMTA03_ordinal',
       'APASIU01_ordinal', 'APBSIU01_ordinal', 'LIPA08_ordinal','CHMA09_ordinal']



# Make all datasets with the same columns and shape
def preprocess_dataset(df, common_df, required_columns):
    # 1. Remove specified columns if they exist
    columns_to_remove = ['EarlyCHD', 'MENOPS01', 'DIABTS02', 'DIABTS03', 'TOTCHOL_V1']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    
    # 2. Add new columns from common dataset
    df = df.merge(common_df[['ID_C', 'HYPTMD01', 'TOTCAL03', 'CURSMK01','DIABTS', 'ANTICOAGCODE01', 'ASPIRINCODE01', 'STATINCODE01']], 
                  on='ID_C', how='left')
    
    # 3. Ensure all required columns are present
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 4. Return the preprocessed dataset
    return df
    
# Imputation process simple/and complex
def impute_data(df, cont_vars, nom_vars, ord_vars, iterative_threshold=1):
    # Combine all variables
    all_vars = cont_vars + nom_vars + ord_vars

    # Separate variables based on missing percentage
    missing_percentages = df[all_vars].isnull().mean()
    high_missing = missing_percentages[missing_percentages > iterative_threshold].index.tolist()
    low_missing = missing_percentages[missing_percentages <= iterative_threshold].index.tolist()

    # Impute MENOPS01 separately for women
    if 'MENOPS01' in df.columns and 'GENDER' in df.columns:
        women_mode = df[df['GENDER'] == 1]['MENOPS01'].mode()[0]
        df.loc[df['GENDER'] == 1, 'MENOPS01'] = df.loc[df['GENDER'] == 1, 'MENOPS01'].fillna(women_mode)

    # Impute categorical variables with mode
    cat_vars = [var for var in nom_vars + ord_vars if var in low_missing]
    if cat_vars:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_vars] = cat_imputer.fit_transform(df[cat_vars])

    cont_vars_low = [var for var in cont_vars if var in low_missing]
    if cont_vars_low:
        cont_imputer = SimpleImputer(strategy='median')
        df[cont_vars_low] = cont_imputer.fit_transform(df[cont_vars_low])

    # Use IterativeImputer for variables with high missing rates
    if high_missing:
        iter_imputer = IterativeImputer(random_state=0)
        df[high_missing] = iter_imputer.fit_transform(df[high_missing])

    return df
    
# Discritization function
def discretize_continuous_variables(df, continuous_vars, n_bins=5):
    """
    Discretize continuous variables into intervals based on quantiles.
    
    Parameters:
    df (pd.DataFrame): The input dataset
    continuous_vars (list): List of continuous variable names to discretize
    n_bins (int): Number of intervals to create (default is 5)
    
    Returns:
    pd.DataFrame: A new dataset with discretized variables
    """
    df_discretized = df.copy()
    bins_dict = {}
    
    for var in continuous_vars:
        if var in df.columns:
            # Create bin edges based on quantiles
            bin_edges = pd.qcut(df[var], q=n_bins, duplicates='drop', retbins=True)[1]
            bins_dict[var] = bin_edges
            
            # Discretize the variable
            df_discretized[f"{var}_ordinal"] = pd.cut(df[var], bins=bin_edges, labels=False, include_lowest=True)
            
            # Add 1 to shift the range from 0-4 to 1-5
            df_discretized[f"{var}_ordinal"] += 1
            
            # Drop the original continuous variable
            df_discretized = df_discretized.drop(columns=[var])
        else:
            print(f"Warning: Variable '{var}' not found in the dataset.")
    
    return df_discretized, bins_dict



    
########################
###### Evaluation ######
########################
def evaluate_and_visualize_model(model, X_test, y_test,fig =True, **predict_params):
    """
    Evaluate model performance with ROC, PR curves, confusion matrices,
    and show the best F1 threshold on the PR plot.
    """
    # Get predictions and probabilities
    try:
        y_pred = model.predict(X_test, **predict_params)
        y_pred_proba = model.predict_proba(X_test, **predict_params)[:, 1]
    except TypeError:
        # Fallback for standard models
        print("Warning: Model does not accept custom predict params. Using default.")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate confusion matrix (default threshold=0.5)
    cm_default = confusion_matrix(y_test, y_pred)
    


    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Compute F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Find the threshold with the highest F1
    optimal_idx = np.argmax(f1_scores)
    # If the optimal index is the last point, threshold can be artificially 1.0,
    # but we check if it is within the length of pr_thresholds
    if optimal_idx < len(pr_thresholds):
        optimal_threshold = pr_thresholds[optimal_idx]
    else:
        optimal_threshold = 1.0
    
    best_f1_score = f1_scores[optimal_idx]

    # Calculate confusion matrix with the optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    

    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    if fig:

            # Print default threshold confusion matrix
        print("\nConfusion Matrix (Default Threshold = 0.5):")
        print("                 Predicted")
        print("                 Class 0  Class 1")
        print(f"Actual Class 0:  {cm_default[0,0]:6d}  {cm_default[0,1]:6d}")
        print(f"Actual Class 1:  {cm_default[1,0]:6d}  {cm_default[1,1]:6d}")
            
        # Print optimal threshold confusion matrix
        print(f"\nConfusion Matrix (Optimal Threshold = {optimal_threshold:.6f}):")
        print("                 Predicted")
        print("                 Class 0  Class 1")
        print(f"Actual Class 0:  {cm_optimal[0,0]:6d}  {cm_optimal[0,1]:6d}")
        print(f"Actual Class 1:  {cm_optimal[1,0]:6d}  {cm_optimal[1,1]:6d}")
    
        # Create ROC and PR curves plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', color='blue')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_title('ROC Curve')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend()
        ax1.grid(True)
    
        # PR Curve
        ax2.plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})', color='green')
        
        # Plot the point for the best F1 threshold
        ax2.scatter(recall[optimal_idx], precision[optimal_idx],
                    color='red',
                    label=(f'Optimal F1 = {best_f1_score:.4f}\n'
                           f'Threshold = {optimal_threshold:.4f}'))
        
        ax2.set_title('Precision-Recall Curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend()
        ax2.grid(True)
    
        plt.tight_layout()
        plt.show()
    
        # Print out the best F1 score
        print(f"\nBest F1 Score from PR curve: {best_f1_score:.4f}")

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_threshold,
        'best_f1_score': best_f1_score,
        'default_cm': cm_default,
        'optimal_cm': cm_optimal}


# Cross Validation functions
def _fit_fold(model, X, y, train_idx, val_idx, fold):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    proba_val = model.predict_proba(X_val)[:, 1]

    # Per-fold optimal threshold
    prec_f, rec_f, thr_f = precision_recall_curve(y_val, proba_val)
    f1_f = 2 * (prec_f * rec_f) / (prec_f + rec_f + 1e-8)
    opt_idx_f = np.argmax(f1_f)
    opt_thr_f = thr_f[opt_idx_f] if opt_idx_f < len(thr_f) else 1.0

    return {
        "fold"     : fold,
        "val_idx"  : val_idx,
        "proba_val": proba_val,
        "roc_auc"  : roc_auc_score(y_val, proba_val),
        "pr_auc"   : average_precision_score(y_val, proba_val),
        "f1"       : f1_score(y_val, (proba_val >= opt_thr_f).astype(int)),
        "threshold": opt_thr_f,
    }


def evaluate_model_cv(model, X, y, cv=5, n_jobs=-1):
    X = np.array(X)
    y = np.array(y)

    skf     = StratifiedKFold(n_splits=cv, shuffle=False)
    splits  = list(skf.split(X, y))

    # 1. Parallel fold execution 
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(model, X, y, train_idx, val_idx, fold)
        for fold, (train_idx, val_idx) in enumerate(splits, 1)
    )
    results = sorted(results, key=lambda x: x["fold"])

    # 2. Collect per-fold results 
    probas_oof     = np.zeros(len(y))
    fold_roc_aucs  = []
    fold_pr_aucs   = []
    fold_f1s       = []
    fold_thresholds = []

    for r in results:
        probas_oof[r["val_idx"]] = r["proba_val"]
        fold_roc_aucs.append(r["roc_auc"])
        fold_pr_aucs.append(r["pr_auc"])
        fold_f1s.append(r["f1"])
        fold_thresholds.append(r["threshold"])
        print(f"  Fold {r['fold']} | ROC-AUC: {r['roc_auc']:.4f} | "
              f"PR-AUC: {r['pr_auc']:.4f} | F1: {r['f1']:.4f} | "
              f"Threshold: {r['threshold']:.4f}")

    # 3. Per-fold averages 
    print(f"\n  Avg ROC-AUC : {np.mean(fold_roc_aucs):.4f} ± {np.std(fold_roc_aucs):.4f}")
    print(f"  Avg PR-AUC  : {np.mean(fold_pr_aucs):.4f} ± {np.std(fold_pr_aucs):.4f}")
    print(f"  Avg F1      : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    #  4. Global OOF evaluation 
    precision, recall, pr_thresholds = precision_recall_curve(y, probas_oof)
    f1_scores   = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 1.0

    y_pred_optimal = (probas_oof >= optimal_threshold).astype(int)
    roc_auc = roc_auc_score(y, probas_oof)
    pr_auc  = average_precision_score(y, probas_oof)
    f1_opt  = f1_score(y, y_pred_optimal)
    cm      = confusion_matrix(y, y_pred_optimal)

    print(f"\n  OOF ROC-AUC : {roc_auc:.4f}")
    print(f"  OOF PR-AUC  : {pr_auc:.4f}")
    print(f"  OOF F1      : {f1_opt:.4f} (threshold={optimal_threshold:.4f})")
    print(f"  Confusion Matrix:\n{cm}")

    # 5. Return everything 
    return {
        # OOF global
        "probas_oof"       : probas_oof,
        "precision"        : precision,
        "recall"           : recall,
        "pr_thresholds"    : pr_thresholds,
        "optimal_threshold": optimal_threshold,
        "roc_auc"          : roc_auc,
        "pr_auc"           : pr_auc,
        "f1_optimal"       : f1_opt,
        "confusion_matrix" : cm,
        "y_pred_optimal"   : y_pred_optimal,
        # Per-fold results
        "fold_roc_aucs"    : fold_roc_aucs,
        "fold_pr_aucs"     : fold_pr_aucs,
        "fold_f1s"         : fold_f1s,
        "fold_thresholds"  : fold_thresholds,
        "mean_roc_auc"     : np.mean(fold_roc_aucs),
        "mean_pr_auc"      : np.mean(fold_pr_aucs),
        "mean_f1"          : np.mean(fold_f1s),
        "std_roc_auc"      : np.std(fold_roc_aucs),
        "std_pr_auc"       : np.std(fold_pr_aucs),
        "std_f1"           : np.std(fold_f1s),
    }

def evaluate_and_compare_models(
    models,
    model_names,
    X_train,
    y_train,
    X_test,
    y_test,
    special_model_idx=None,
    special_predict_params=None,
    plot_train_curves=True,
    plot_gap=True,
    title_roc="ROC Curve Comparison",
    title_pr="PR-AUC Comparison",
    title_gap="Generalization Gap",

    round_digits=4,
    color_palette="muted",
    figsize=None,
    width_per_panel=6,
    show=True,
    save_path=None,
    save_name="model_comparison.png",
    save_dpi=300,
    save_format="png",
    bbox_inches="tight",
    transparent=False
):
    """
    Evaluate and compare multiple models on the same train/test data with ROC, PR curves,
    and optional overfitting gap analysis.

    Parameters:
    models : list
        List of fitted models with predict and predict_proba methods.
    model_names : list of str
        Names for each model.
    X_train, y_train, X_test, y_test : array-like
        Training and test data.
    special_model_idx : int, optional
        Index of model that requires special predict_proba parameters.
    special_predict_params : dict, optional
        Parameters to pass to predict_proba for the special model.
    plot_train_curves : bool, default=True
        Whether to plot training curves.
    plot_gap : bool, default=True
        Whether to plot the gap analysis subplot.
    title_roc : str, default="ROC Curve Comparison"
        Title for ROC plot.
    title_pr : str, default="PR-AUC Comparison"
        Title for PR plot.
    round_digits : int, default=4
        Number of decimal digits used in printed outputs and summary table.
    color_palette : str, default="muted"
        Matplotlib/seaborn-style palette name, or a list of colors.
    figsize : tuple, optional
        Figure size. Auto-chosen if None.
    show : bool, default=True
        Whether to display the figure.
    """

    if len(models) != len(model_names):
        raise ValueError("Number of models must match number of model names")

    try:
        import seaborn as sns
        if isinstance(color_palette, str):
            colors = sns.color_palette(color_palette, n_colors=len(models))
        else:
            colors = color_palette
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20")
        colors = [cmap(i / max(len(models) - 1, 1)) for i in range(len(models))]

    def get_predictions(X, model, model_idx, is_proba=False):
        if model_idx == special_model_idx and special_predict_params is not None and is_proba:
            try:
                y_pred_proba = model.predict_proba(X, **special_predict_params)[:, 1]
                y_pred = model.predict(X)
                return y_pred, y_pred_proba
            except TypeError:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1]
                return y_pred, y_pred_proba
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        return y_pred, y_pred_proba

    ncols = 2 + int(plot_gap)

    if figsize is None:
        figsize = (width_per_panel * ncols, 5)

    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    if ncols == 2:
        ax1, ax2 = axes
        ax3 = None
    else:
        ax1, ax2, ax3 = axes

    results = {}
    gap_data = {'model_names': [], 'roc_auc_gaps': [], 'pr_auc_gaps': [], 'f1_gaps': []}
    summary_data = []

    print("=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    for idx, (model, name) in enumerate(zip(models, model_names)):
        color = colors[idx]

        y_train_pred, y_train_proba = get_predictions(X_train, model, idx, is_proba=True)
        y_test_pred, y_test_proba = get_predictions(X_test, model, idx, is_proba=True)

        cm_train_default = confusion_matrix(y_train, y_train_pred)
        precision_train, recall_train, pr_thresholds_train = precision_recall_curve(y_train, y_train_proba)
        f1_scores_train = 2 * (precision_train * recall_train) / (precision_train + recall_train + 1e-8)
        optimal_idx_train = np.argmax(f1_scores_train)
        optimal_threshold_train = pr_thresholds_train[optimal_idx_train] if optimal_idx_train < len(pr_thresholds_train) else 1.0
        best_f1_train = f1_scores_train[optimal_idx_train]
        y_train_pred_optimal = (y_train_proba >= optimal_threshold_train).astype(int)
        cm_train_optimal = confusion_matrix(y_train, y_train_pred_optimal)

        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        roc_auc_train = auc(fpr_train, tpr_train)
        pr_auc_train = average_precision_score(y_train, y_train_proba)

        cm_test_default = confusion_matrix(y_test, y_test_pred)
        precision_test, recall_test, pr_thresholds_test = precision_recall_curve(y_test, y_test_proba)
        f1_scores_test = 2 * (precision_test * recall_test) / (precision_test + recall_test + 1e-8)
        optimal_idx_test = np.argmax(f1_scores_test)
        optimal_threshold_test = pr_thresholds_test[optimal_idx_test] if optimal_idx_test < len(pr_thresholds_test) else 1.0
        best_f1_test = f1_scores_test[optimal_idx_test]
        y_test_pred_optimal = (y_test_proba >= optimal_threshold_test).astype(int)
        cm_test_optimal = confusion_matrix(y_test, y_test_pred_optimal)

        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
        roc_auc_test = auc(fpr_test, tpr_test)
        pr_auc_test = average_precision_score(y_test, y_test_proba)

        tn_test, fp_test, fn_test, tp_test = cm_test_optimal.ravel()
        sensitivity_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0
        specificity_test = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0
        precision_test_opt = tp_test / (tp_test + fp_test) if (tp_test + fp_test) > 0 else 0
        npv_test = tn_test / (tn_test + fn_test) if (tn_test + fn_test) > 0 else 0
        accuracy_test = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test)

        tn_train, fp_train, fn_train, tp_train = cm_train_optimal.ravel()
        sensitivity_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0
        specificity_train = tn_train / (tn_train + fp_train) if (tn_train + fp_train) > 0 else 0
        accuracy_train = (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)

        roc_gap = roc_auc_train - roc_auc_test
        pr_gap = pr_auc_train - pr_auc_test
        f1_gap = best_f1_train - best_f1_test
        sensitivity_gap = sensitivity_train - sensitivity_test
        specificity_gap = specificity_train - specificity_test
        accuracy_gap = accuracy_train - accuracy_test

        gap_data['model_names'].append(name)
        gap_data['roc_auc_gaps'].append(roc_gap)
        gap_data['pr_auc_gaps'].append(pr_gap)
        gap_data['f1_gaps'].append(f1_gap)

        summary_data.append({
            'Model': name,
            'Optimal_Threshold': optimal_threshold_test,
            'ROC_AUC': roc_auc_test,
            'PR_AUC': pr_auc_test,
            'F1_Score': best_f1_test,
            'Sensitivity_Recall': sensitivity_test,
            'Specificity': specificity_test,
            'Precision': precision_test_opt,
            'NPV': npv_test,
            'Accuracy': accuracy_test,
            'TP': tp_test,
            'TN': tn_test,
            'FP': fp_test,
            'FN': fn_test,
            'ROC_AUC_Gap': roc_gap,
            'PR_AUC_Gap': pr_gap,
            'F1_Gap': f1_gap,
            'Sensitivity_Gap': sensitivity_gap,
            'Specificity_Gap': specificity_gap,
            'Accuracy_Gap': accuracy_gap
        })

        fmt = f".{round_digits}f"
        print(f"\n{'-' * 80}")
        print(f"MODEL: {name}")
        if idx == special_model_idx:
            print(f"(Using special parameters: {special_predict_params})")
        print(f"{'-' * 80}")
        print(f"{'Metric':<25} {'Train':<15} {'Test':<15} {'Gap':<15}")
        print("-" * 70)
        print(f"{'ROC AUC':<25} {format(roc_auc_train, fmt):<15} {format(roc_auc_test, fmt):<15} {format(roc_gap, f'+.{round_digits}f')}")
        print(f"{'PR AUC':<25} {format(pr_auc_train, fmt):<15} {format(pr_auc_test, fmt):<15} {format(pr_gap, f'+.{round_digits}f')}")
        print(f"{'Best F1 Score':<25} {format(best_f1_train, fmt):<15} {format(best_f1_test, fmt):<15} {format(f1_gap, f'+.{round_digits}f')}")
        print(f"{'Sensitivity (Recall)':<25} {format(sensitivity_train, fmt):<15} {format(sensitivity_test, fmt):<15} {format(sensitivity_gap, f'+.{round_digits}f')}")
        print(f"{'Specificity':<25} {format(specificity_train, fmt):<15} {format(specificity_test, fmt):<15} {format(specificity_gap, f'+.{round_digits}f')}")
        print(f"{'Accuracy':<25} {format(accuracy_train, fmt):<15} {format(accuracy_test, fmt):<15} {format(accuracy_gap, f'+.{round_digits}f')}")
        print(f"{'Optimal Threshold':<25} {format(optimal_threshold_train, f'.{round_digits}f'):<15} {format(optimal_threshold_test, f'.{round_digits}f'):<15} {format(optimal_threshold_train - optimal_threshold_test, f'+.{round_digits}f')}")

        ax1.plot(fpr_test, tpr_test, label=f'{name} ({roc_auc_test:.{round_digits}f})', color=color, linewidth=2.2)
        if plot_train_curves:
            ax1.plot(fpr_train, tpr_train, linestyle='--', color=color, linewidth=1.8, alpha=0.7)

        ax2.plot(recall_test, precision_test, label=f'{name} ({pr_auc_test:.{round_digits}f})', color=color, linewidth=2.2)
        if plot_train_curves:
            ax2.plot(recall_train, precision_train, linestyle='--', color=color, linewidth=1.8, alpha=0.7)

        ax2.scatter(recall_test[optimal_idx_test], precision_test[optimal_idx_test],
                    color=color, s=50, marker='s', zorder=5)

        if plot_train_curves:
            ax2.scatter(recall_train[optimal_idx_train], precision_train[optimal_idx_train],
                        color=color, s=45, marker='o', alpha=0.5, zorder=5)

        results[name] = {
            'train': {
                'roc_auc': roc_auc_train,
                'pr_auc': pr_auc_train,
                'optimal_threshold': optimal_threshold_train,
                'best_f1_score': best_f1_train,
                'sensitivity': sensitivity_train,
                'specificity': specificity_train,
                'accuracy': accuracy_train,
                'default_cm': cm_train_default,
                'optimal_cm': cm_train_optimal
            },
            'test': {
                'roc_auc': roc_auc_test,
                'pr_auc': pr_auc_test,
                'optimal_threshold': optimal_threshold_test,
                'best_f1_score': best_f1_test,
                'sensitivity': sensitivity_test,
                'specificity': specificity_test,
                'precision': precision_test_opt,
                'npv': npv_test,
                'accuracy': accuracy_test,
                'default_cm': cm_test_default,
                'optimal_cm': cm_test_optimal
            },
            'gaps': {
                'roc_auc_gap': roc_gap,
                'pr_auc_gap': pr_gap,
                'f1_gap': f1_gap,
                'sensitivity_gap': sensitivity_gap,
                'specificity_gap': specificity_gap,
                'accuracy_gap': accuracy_gap
            }
        }

    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)
    ax1.set_title(title_roc, fontsize=12, fontweight='bold')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    
    baseline = np.mean(y_test)
    ax2.axhline(
        y=baseline,
        color='gray',
        linestyle='--',
        linewidth=1.2,
        alpha=0.8,
        label=f'Baseline ({baseline:.2f})'
    )
    ax2.set_title(title_pr, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    if plot_gap and ax3 is not None:
        x_pos = np.arange(len(gap_data['model_names']))
        width = 0.25

        bars1 = ax3.bar(x_pos - width, gap_data['roc_auc_gaps'], width, label='ROC AUC Gap', color='steelblue', alpha=0.75)
        bars2 = ax3.bar(x_pos, gap_data['pr_auc_gaps'], width, label='PR AUC Gap', color='sandybrown', alpha=0.75)
        bars3 = ax3.bar(x_pos + width, gap_data['f1_gaps'], width, label='F1 Score Gap', color='seagreen', alpha=0.75)

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Train - Test Gap')
        ax3.set_title(title_gap, fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(gap_data['model_names'], rotation=45, ha='right')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.{round_digits}f}',
                    ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontsize=7
                )

    plt.tight_layout()

    if save_path is None:
        if "__file__" in globals():
            base_dir = Path(__file__).resolve().parent
        else:
            base_dir = Path.cwd()   # works in Jupyter
        save_path = base_dir / f"{save_name}.{save_format}"
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        save_path,
        dpi=save_dpi,
        format=save_format,
        bbox_inches=bbox_inches,
        transparent=transparent
    )
    
    if show:
        plt.show()

    summary_df = pd.DataFrame(summary_data)

    column_order = [
        'Model', 'Optimal_Threshold',
        'ROC_AUC', 'PR_AUC', 'F1_Score',
        'Sensitivity_Recall', 'Specificity', 'Precision', 'NPV', 'Accuracy',
        'TP', 'TN', 'FP', 'FN',
        'ROC_AUC_Gap', 'PR_AUC_Gap', 'F1_Gap',
        'Sensitivity_Gap', 'Specificity_Gap', 'Accuracy_Gap'
    ]
    summary_df = summary_df[column_order].round(round_digits)

    print("\n" + "=" * 80)
    print("OVERALL COMPARISON (Test Set at Optimal Threshold)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    return results, summary_df






####################################
###### Filteration Functions #######
####################################

def count_feature_occurrences(rules_df, feature_set, rule_col_name = 'rule'):
    feature_counts = {feature: 0 for feature in feature_set}
    
    # Convert to list if it's a pandas Series
    rules = rules_df[rule_col_name].tolist()
    
    for rule in rules:
        for feature in feature_set:
            # Use a regular expression to find the feature as a whole word
            if feature in str(rule):
                feature_counts[feature] += 1
    return feature_counts

def filter_df_by_string(rules_df, search_keywords, column_name='rule',
                        match='any', active_only=False,
                        group_by=False, sort_by='abs_coef',
                        X=None, y=None, rule_col_width=600):
    """
    Filter a rules DataFrame by one or more keyword strings.

    Parameters
    rules_df        : pd.DataFrame from model.get_rules()
    search_keywords : str or list of str
    column_name     : column to search in (default: 'rule')
    match           : 'any' → OR logic (default), 'all' → AND logic
    active_only     : if True, pre-filter to non-zero coefficients only
    group_by        : if True, rows grouped by matched_keywords, then sort_by
    sort_by         : column to sort by descending (default: 'abs_coef')
                      pass None to skip sorting
    X               : pd.DataFrame — training features to compute support
    y               : pd.Series/array — target labels (0/1) aligned with X
                      requires X to be provided; adds class distribution columns
    rule_col_width  : int, pixel width of the rule column in display (default: 600)

    Returns    pd.DataFrame of matching rows
    """
    if not isinstance(search_keywords, (str, list)):
        raise TypeError("search_keywords must be a str or list of str")
    if isinstance(search_keywords, str):
        search_keywords = [search_keywords]
    if y is not None and X is None:
        raise ValueError("X must be provided when y is provided")

    df = rules_df.copy()

    if active_only and 'coefficient' in df.columns:
        df = df[df['coefficient'].abs() > 1e-6]

    #  Filter mask 
    masks = [df[column_name].str.contains(kw, na=False, regex=False)
             for kw in search_keywords]

    if match == 'any':
        mask = masks[0]
        for m in masks[1:]: mask = mask | m
    elif match == 'all':
        mask = masks[0]
        for m in masks[1:]: mask = mask & m
    else:
        raise ValueError("match must be 'any' or 'all'")

    result = df[mask].reset_index(drop=False).rename(columns={'index': 'original_idx'})

    #  Matched keywords column 
    result['matched_keywords'] = result[column_name].apply(
        lambda rule_str: ', '.join([kw for kw in search_keywords if kw in str(rule_str)])
    )

    # Support + target distribution 
    if X is not None:
        import numpy as np
        n_samples = len(X)
        y_arr = np.array(y) if y is not None else None

        support_counts, support_pcts = [], []
        class0_counts, class0_pcts   = [], []
        class1_counts, class1_pcts   = [], []

        for rule in result[column_name]:
            try:
                fired = X.eval(rule).fillna(False).astype(bool)
                count = int(fired.sum())
            except Exception as e:
                print(f"  ⚠️ Could not evaluate rule: '{rule}'\n     {e}")
                support_counts.append(-1); support_pcts.append(None)
                class0_counts.append(None); class0_pcts.append(None)
                class1_counts.append(None); class1_pcts.append(None)
                continue

            support_counts.append(count)
            support_pcts.append(round(count / n_samples * 100, 2))

            if y_arr is not None and count > 0:
                y_fired  = y_arr[fired.values]
                c1       = int(y_fired.sum())
                c0       = count - c1
                class1_counts.append(c1)
                class0_counts.append(c0)
                class1_pcts.append(round(c1 / count * 100, 2))
                class0_pcts.append(round(c0 / count * 100, 2))
            else:
                class0_counts.append(None); class0_pcts.append(None)
                class1_counts.append(None); class1_pcts.append(None)

        result['support_count'] = support_counts
        result['support_pct']   = support_pcts

        if y_arr is not None:
            result['class0_count'] = class0_counts
            result['class0_pct']   = class0_pcts   # % of samples that are class 0
            result['class1_count'] = class1_counts
            result['class1_pct']   = class1_pcts   # % of samples that are class 1

    #  Sort 
    if sort_by is not None:
        if sort_by not in result.columns:
            raise ValueError(f"sort_by='{sort_by}' not found in columns: {result.columns.tolist()}")
        if group_by:
            result = result.sort_values(
                ['matched_keywords', sort_by],
                ascending=[True, False]
            ).reset_index(drop=True)
        else:
            result = result.sort_values(sort_by, ascending=False).reset_index(drop=True)

    #  Display with wide rule column
    styled = (
        result.style
        .set_properties(subset=[column_name],
                        **{'min-width': f'{rule_col_width}px',
                           'white-space': 'pre-wrap',
                           'text-align': 'left'})
        .set_properties(subset=[c for c in result.columns if c != column_name],
                        **{'text-align': 'center'})
        .format({
            'support_pct' : '{:.1f}%',
            'class0_pct'  : '{:.1f}%',
            'class1_pct'  : '{:.1f}%',
            'abs_coef'    : '{:.4f}',
            'coefficient' : '{:.4f}',
        }, na_rep='-')
    )

    print(f"Found {len(result)} rules [{match.upper()} match] for: {search_keywords}")
    display(styled)
    return result

####################################
######## Analysis Functions ########
####################################
#process_dataframe_V2 to process_dataframe
# This function ment to process the recommendation dataset from binned values to continious (must use the same extracted bin_edges)
def process_dataframe(df, org_df, bin_edges, round_to=2):
    df_to_process = df.copy()
    
    # Function to round values if round_to is specified
    def round_value(x):
        if round_to is not None and pd.notnull(x):
            return round(x, round_to)
        return x
    
    # Apply rounding to all numeric columns
    numeric_columns = df_to_process.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df_to_process[col] = df_to_process[col].apply(round_value)
    
    # Step 1: Apply np.ceil to Min and np.floor to Max
    df_to_process['Min'] = pd.to_numeric(df_to_process['Min'], errors='coerce')
    df_to_process['Max'] = pd.to_numeric(df_to_process['Max'], errors='coerce')
    df_to_process['Min'] = np.ceil(df_to_process['Min'])   # keep float, no .astype(int)
    df_to_process['Max'] = np.floor(df_to_process['Max'])  # keep float, no .astype(int)

    # Step 2: Convert Min and Max to continuous intervals using discretization thresholds
    for feature, thresholds in bin_edges.items():
        feature_name = feature + '_ordinal'
        if feature_name in df_to_process['Feature'].values:
            feature_rows = df_to_process['Feature'] == feature_name
            
            # Replace Min values with the lower bound of the corresponding interval
            df_to_process.loc[feature_rows, 'Min'] = df_to_process.loc[feature_rows, 'Min'].apply(
                lambda x: round_value(thresholds[int(x) - 1]) if 1 <= x <= len(thresholds) else np.nan
            ).astype(float)
            
            # Ensure Min is not less than the minimum value in original data
            min_value = round_value(org_df[feature].min())
            df_to_process.loc[feature_rows, 'Min'] = df_to_process.loc[feature_rows, 'Min'].clip(lower=min_value)

            # Replace Max values with the upper bound of the corresponding interval
            df_to_process.loc[feature_rows, 'Max'] = df_to_process.loc[feature_rows, 'Max'].apply(
                lambda x: round_value(thresholds[int(x)]) if 1 <= x < len(thresholds) else np.nan
            ).astype(float)
            
            # Ensure Max is not greater than the maximum value in orginal data
            max_value = round_value(org_df[feature].max())
            df_to_process.loc[feature_rows, 'Max'] = df_to_process.loc[feature_rows, 'Max'].clip(upper=max_value)

    # Step 3: Update Current values using org_df based on patient IDs
    for i, row in df_to_process.iterrows():
        patient_id = row['Patient_IDs']
        feature = row['Feature'].replace('_ordinal', '')
        if patient_id in org_df.index and feature in org_df.columns:
            df_to_process.at[i, 'Current'] = round_value(org_df.loc[patient_id, feature])

    # Step 4: Update Rounded Recommended based on Current, Min, and Max values
    for feature, thresholds in bin_edges.items():
        feature_name = feature + '_ordinal'
        if feature_name in df_to_process['Feature'].values:
            feature_rows = df_to_process['Feature'] == feature_name
            df_to_process.loc[feature_rows, 'Rounded Recommended'] = df_to_process.loc[feature_rows, 'Rounded Recommended'].apply(
                lambda x: round_value(thresholds[int(x) - 1]) if pd.notnull(x) and 1 <= x <= len(thresholds) else np.nan
            )
    # Step 5: Calculate Difference based on specified logic
    df_to_process['Difference'] = df_to_process.apply(
            lambda row: (
                round_value(row['Max'] - row['Current'])
                if pd.notnull(row['Difference']) and row['Difference'] < 0
                else round_value(row['Min'] - row['Current'])
            ),
            axis=1
        )
            
    # Remove '_ordinal' suffix from all feature names
    df_to_process['Feature'] = df_to_process['Feature'].str.replace('_ordinal', '', regex=False)

    return df_to_process

def process_value_intervals_floor_ceil(value):
    """Process individual feature values with robust parsing"""
    try:
        # Maintain 'No change' entries
        if value in ('No change', '((No change-empty rec))'):
            return value
        
        # Handle string representations of lists
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    value = parsed
            except:
                pass
        
        # Process only valid 2-element lists
        if isinstance(value, list) and len(value) == 2:
            # Convert elements to floats safely
            try:
                float1 = float(value[0])
                float2 = float(value[1])
            except (TypeError, ValueError) as e:
                print(f"Invalid numeric values in {value}: {str(e)}")
                return value
            
            # Apply rounding rules
            if float2 > float1:
                return math.ceil(float1)
            elif float2 < float1:
                return math.floor(float1)
            return float1  # Equal values
        
        return value  # Return original for non-processed cases
        
    except Exception as e:
        print(f"Error processing {value}: {str(e)}")
        return value

def preprocess_features_intervals(df, feature_columns):
    
    """Process feature columns according to specified rounding rules"""
            
    processed_df = df.copy()
    
    for col in feature_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].apply(process_value_intervals_floor_ceil)
    
    return processed_df


def generate_report(df, patient_id):
    """Generate recommendation report — reads Final Recommendation as DataFrame or string."""

    required_columns = {'Patient_ID', 'Final Recommendation'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    empty_df = pd.DataFrame(columns=['Feature', 'Min', 'Max', 'Current', 'Recommended',
                                     'Rounded Recommended', 'Difference', 'Org Risk',
                                     'Partial Risk', 'Risk Decreased'])

    patient_data = df[df['Patient_ID'] == patient_id]
    if patient_data.empty:
        return empty_df

    mode_val = patient_data['mode'].iloc[0] if 'mode' in patient_data.columns and pd.notnull(patient_data['mode'].iloc[0]) else 'Unknown'
    recommendation = patient_data['Final Recommendation'].iloc[0]

    # Check for None / NaN / empty
    if recommendation is None or (isinstance(recommendation, float) and np.isnan(recommendation)):
        print(f"Patient ID {patient_id} in mode '{mode_val}' has no recommendations.")
        return empty_df

    # Check for known string markers indicating no recommendation
    if isinstance(recommendation, str):
        no_rec_markers = [
            'No Rec Apply', 'No change', '((No change-empty rec))',
            'No Feasible Recommendation', 'No selection made',
            'Job did not complete', 'Job Failed'
        ]
        if any(marker in recommendation for marker in no_rec_markers) or '\n\n' not in recommendation:
            print(f"Patient ID {patient_id} in mode '{mode_val}' has no recommendations.")
            return empty_df

    try:
        # Case 1: Already a DataFrame (new path from validate_patient_features) 
        if isinstance(recommendation, pd.DataFrame):
            if recommendation.empty:
                print(f"Patient ID {patient_id} in mode '{mode_val}' has no recommendations.")
                return empty_df
            final_df = recommendation.copy()

        # Case 2: String representation (legacy path) 
        elif isinstance(recommendation, str):
            parts_split = recommendation.split('\n\n', 1)
            if len(parts_split) < 2:
                print(f"Patient ID {patient_id} in mode '{mode_val}' has no recommendations.")
                return empty_df
            main_part, diff_part = parts_split

            main_data = []
            for line in main_part.split('\n'):
                line = line.strip()
                if not line or line.startswith('Feature'):
                    continue
                clean_line = re.sub(r'^\d+\s+', '', line)
                parts = re.split(r'\s+', clean_line)
                if len(parts) >= 6:
                    main_data.append({
                        'Feature': parts[0],
                        'Min': parts[1],
                        'Max': parts[2],
                        'Current': parts[3],
                        'Recommended': parts[4],
                        'Rounded Recommended': parts[5]
                    })

            diff_data = []
            for line in diff_part.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    parts = re.split(r'\s+', re.sub(r'^\d+\s*', '', line))
                    if len(parts) == 4:
                        diff_data.append({
                            'Difference': parts[0],
                            'Org Risk': parts[1],
                            'Partial Risk': parts[2],
                            'Risk Decreased': parts[3]
                        })

            if not main_data:
                print(f"Patient ID {patient_id} in mode '{mode_val}' has no recommendations.")
                return empty_df

            final_df = pd.concat([pd.DataFrame(main_data), pd.DataFrame(diff_data)], axis=1)

        else:
            print(f"Patient ID {patient_id} in mode '{mode_val}' has no recommendations.")
            return empty_df

        # Shared numeric conversion 
        if 'Min' in final_df.columns and 'Max' in final_df.columns:
            final_df[['Min', 'Max']] = final_df[['Min', 'Max']].replace({
                'inf': float('inf'), '-inf': -float('inf')
            })

        numeric_cols = [c for c in ['Min', 'Max', 'Current', 'Recommended',
                                    'Rounded Recommended', 'Difference',
                                    'Org Risk', 'Partial Risk'] if c in final_df.columns]
        final_df[numeric_cols] = final_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        return final_df.dropna()

    except Exception as e:
        print(f"Patient ID {patient_id} in mode '{mode_val}' has no recommendations. (Error detail: {e})")
        return empty_df
        

def extract_partials_modifications(df):
    patient_ID_list = []
    Feature_partial_list = []
    Maxs_list = []
    Mins_list = []
    Org_risks_list = []
    Current_val_list =[]
    Recommended_list = []
    Differences_list = []
    Partial_risk_list = []
    is_Reduced_risk_partial = []

    for i in range(len(df)):
        patient_id = df['Patient_ID'].iloc[i]
        recommendation_df = generate_report(df.iloc[[i]], patient_id)
        for j in range(len(recommendation_df)):
            patient_ID_list.append(patient_id)
            Feature_partial_list.append(recommendation_df['Feature'].iloc[j])
            Org_risks_list.append(recommendation_df['Org Risk'].iloc[j])
            Current_val_list.append(recommendation_df['Current'].iloc[j])
            Recommended_list.append(recommendation_df['Rounded Recommended'].iloc[j])
            Differences_list.append(recommendation_df['Difference'].iloc[j])
            Partial_risk_list.append(recommendation_df['Partial Risk'].iloc[j])
            is_Reduced_risk_partial.append(recommendation_df['Risk Decreased'].iloc[j])
            # Preserve infinite values to allow correct fallback in truncate_df
            Mins_list.append(recommendation_df['Min'].iloc[j])
            Maxs_list.append(recommendation_df['Max'].iloc[j])

                
            
    
    return pd.DataFrame({
                        'Patient_IDs': patient_ID_list,
                        'Feature': Feature_partial_list,
                        'Min':Mins_list,
                        'Max':Maxs_list,
                        'Current': Current_val_list,
                        'Rounded Recommended': Recommended_list,
                        'Difference': Differences_list,
                        'Org Risk': Org_risks_list,
                        'Partial Risk': Partial_risk_list,
                        'Risk Decreased': is_Reduced_risk_partial
                    })

        
# This is main function in the analyzes as it generate the datasets of "STRR (low)" and "LTRR (high)"
feature_columns = changable_feature_ordinal+changable_feature_ordinal
def generate_df_for_analyzys(Output_df, org_df, bin_edges, healthy_patient_index, Health_Guiedance):

    def truncate_df(df):
        df = df.copy()
        for i in range(len(df)):
            feature        = df.iloc[i, 1]
            
            # Priority 3 (Quantiles)
            temp_min       = round(float(org_df.loc[healthy_patient_index][feature].quantile(0.025)), 2)
            temp_max       = round(float(org_df.loc[healthy_patient_index][feature].quantile(0.975)), 2)
            
            # Priority 4 (Absolute Data Range)
            abs_min_data   = round(float(org_df[feature].min()), 2)
            abs_max_data   = round(float(org_df[feature].max()), 2)
            
            # Initial bounds from GAMS (Priority 1)
            raw_min = df.iloc[i, 2]
            raw_max = df.iloc[i, 3]
            
            f_min = raw_min if pd.notnull(raw_min) and not np.isinf(raw_min) else None
            f_max = raw_max if pd.notnull(raw_max) and not np.isinf(raw_max) else None

            # Get Clinical Bounds (Priority 2)
            guidance_row = Health_Guiedance[Health_Guiedance['Feature'] == feature]
            p2_min, p2_max = None, None
            if not guidance_row.empty:
                g_min = guidance_row.iloc[0]['Min']
                g_max = guidance_row.iloc[0]['Max']
                if pd.notnull(g_min) and not (isinstance(g_min, (float, int)) and np.isinf(float(g_min))):
                    p2_min = round(float(g_min), 2)
                if pd.notnull(g_max) and not (isinstance(g_max, (float, int)) and np.isinf(float(g_max))):
                    p2_max = round(float(g_max), 2)

            # Logic of priority for Min (Priority 2 -> 3 -> 4)
            if f_min is None:
                # Use clinical Min if it's available and doesn't violate fixed Max
                if p2_min is not None and (f_max is None or p2_min < f_max):
                    f_min = p2_min
                elif f_max is None or temp_min < f_max:
                    f_min = temp_min # Priority 3 fallback
                else:
                    f_min = abs_min_data # Final safeguard (Priority 4)

            # Resolve Missing Max (Priority 2 -> 3 -> 4)
            if f_max is None:
                # Use clinical Max if it's available and doesn't violate fixed Min
                if p2_max is not None and (f_min is None or p2_max > f_min):
                    f_max = p2_max
                elif f_min is None or temp_max > f_min:
                    f_max = temp_max # Priority 3 fallback
                else:
                    f_max = abs_max_data # Final safeguard (Priority 4)

            df.iloc[i, 2] = round(float(f_min), 2)
            df.iloc[i, 3] = round(float(f_max), 2)

        return df

    filtered         = Output_df[Output_df['New risk(RB)'] != 'Low-Risk - No Recommendations']
    high_risk_df_org = filtered[filtered['mode'] == 'LTRR'].copy()
    low_risk_df_org  = filtered[filtered['mode'] == 'STRR'].copy()

    high_risk_df_p = preprocess_features_intervals(high_risk_df_org, feature_columns)
    low_risk_df_p  = preprocess_features_intervals(low_risk_df_org,  feature_columns)

    Partial_impact_df_low  = extract_partials_modifications(low_risk_df_p)
    Partial_impact_df_high = extract_partials_modifications(high_risk_df_p)

    Partial_impact_df_low_p  = process_dataframe(Partial_impact_df_low, org_df, bin_edges)
    Partial_impact_df_high_p = process_dataframe(Partial_impact_df_high, org_df, bin_edges)

    Partial_impact_df_low_p_truncated  = truncate_df(Partial_impact_df_low_p)
    Partial_impact_df_high_p_truncated = truncate_df(Partial_impact_df_high_p)

    return (low_risk_df_p, high_risk_df_p,
            Partial_impact_df_low_p, Partial_impact_df_high_p,
            Partial_impact_df_low_p_truncated, Partial_impact_df_high_p_truncated)



def analyze_risks(
    df_strr,
    df_ltrr,
    org_col='Org Risk (RB)',
    new_col='New risk(RB)',
    n_boot=10000,
    alpha=0.05,
    random_state=42,
    plot=True
):
    """
    Paired risk analysis for STRR and LTRR vs the same baseline.
    """

    rng = np.random.default_rng(random_state)

    #  helpers 
    def _clean_numeric(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan,
                       'NA': np.nan, 'N/A': np.nan})
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace('%', '', regex=False)
        return pd.to_numeric(s, errors='coerce')

    def _sign_test_one_sided(diffs: np.ndarray) -> dict:
        d = np.asarray(diffs, dtype=float)
        d = d[np.isfinite(d)]

        pos = int(np.sum(d > 0))
        neg = int(np.sum(d < 0))
        zero = int(np.sum(d == 0))
        n_eff = pos + neg

        if n_eff == 0:
            p = np.nan
        else:
            p = binomtest(pos, n=n_eff, p=0.5, alternative='greater').pvalue

        return {
            'n_positive': pos,
            'n_negative': neg,
            'n_zero': zero,
            'n_effective': n_eff,
            'p_value': float(p) if p == p else np.nan
        }

    def _bootstrap_mean_ci(diffs: np.ndarray):
        d = np.asarray(diffs, dtype=float)
        d = d[np.isfinite(d)]
        mean_d = float(np.mean(d))
        if len(d) < 2:
            return mean_d, np.nan, np.nan
        res = bootstrap(
            (d,),
            np.mean,
            confidence_level=1 - alpha,
            n_resamples=n_boot,
            method='BCa',
            random_state=random_state
        )
        return mean_d, float(res.confidence_interval.low), float(res.confidence_interval.high)

    def _prop_ci(count: int, n: int, z=1.96):
        if n == 0:
            return np.nan, np.nan
        p = count / n
        half = z * np.sqrt(p * (1 - p) / n)
        return max(0.0, p - half), min(1.0, p + half)

    def _summarize_strategy(org, new, red, label: str) -> dict:
        n_total = len(red)
        mean_arr, ci_low, ci_high = _bootstrap_mean_ci(red)
        sign_res = _sign_test_one_sided(red)

        n_red = int(np.sum(red > 0))
        n_eq = int(np.sum(red == 0))
        n_worse = int(np.sum(red < 0))

        prop_red = n_red / n_total
        prop_eq = n_eq / n_total
        prop_worse = n_worse / n_total

        ci_red = _prop_ci(n_red, n_total)
        ci_eq = _prop_ci(n_eq, n_total)
        ci_worse = _prop_ci(n_worse, n_total)

        n_no_reduction = n_eq + n_worse
        pct_reduced = round((n_red / n_total) * 100, 4)
        pct_no_reduction = round((n_no_reduction / n_total) * 100, 4)
        p_red_frac = n_red / n_total
        p_no_red_frac = n_no_reduction / n_total

        ci_reduced_pct = (
            round(max(0.0, p_red_frac - 1.96 * np.sqrt(p_red_frac * (1 - p_red_frac) / n_total)), 4),
            round(min(1.0, p_red_frac + 1.96 * np.sqrt(p_red_frac * (1 - p_red_frac) / n_total)), 4)
        )
        # Confidence interval of NO reduction (this is what was missing/hidden before)
        ci_no_reduction_pct = (
            round(max(0.0, p_no_red_frac - 1.96 * np.sqrt(p_no_red_frac * (1 - p_no_red_frac) / n_total)), 4),
            round(min(1.0, p_no_red_frac + 1.96 * np.sqrt(p_no_red_frac * (1 - p_no_red_frac) / n_total)), 4)
        )

        return {
            'label': label,
            'n': n_total,
            'mean_org': float(np.mean(org)),
            'median_org': float(np.median(org)),
            'std_org': float(np.std(org, ddof=1)) if n_total > 1 else np.nan,
            'mean_new': float(np.mean(new)),
            'median_new': float(np.median(new)),
            'std_new': float(np.std(new, ddof=1)) if n_total > 1 else np.nan,
            'mean_arr': mean_arr,
            'median_arr': float(np.median(red)),
            'bootstrap_ci_low': ci_low,
            'bootstrap_ci_high': ci_high,
            'sign_test_p_one_sided': sign_res['p_value'],
            'n_positive_reduction': sign_res['n_positive'],
            'n_zero_reduction': sign_res['n_zero'],
            'n_negative_reduction': sign_res['n_negative'],
            'prop_reduced': prop_red,
            'prop_equal': prop_eq,
            'prop_worse': prop_worse,
            'prop_reduced_ci_low': ci_red[0],
            'prop_reduced_ci_high': ci_red[1],
            'prop_equal_ci_low': ci_eq[0],
            'prop_equal_ci_high': ci_eq[1],
            'prop_worse_ci_low': ci_worse[0],
            'prop_worse_ci_high': ci_worse[1],
            'percentage_reduced': pct_reduced,
            'percentage_no_reduction': pct_no_reduction,
            'reduced_count': n_red,
            'no_reduction_count': n_no_reduction,
            'ci_reduced': ci_reduced_pct,
            'ci_no_reduction': ci_no_reduction_pct,  
        }

    #  cleaning & alignment 
    df_strr = df_strr.copy()
    df_ltrr = df_ltrr.copy()
    df_strr.columns = df_strr.columns.str.strip()
    df_ltrr.columns = df_ltrr.columns.str.strip()

    for c in (org_col, new_col):
        if c not in df_strr.columns:
            raise KeyError(f"STRR missing column: {c}")
        if c not in df_ltrr.columns:
            raise KeyError(f"LTRR missing column: {c}")

    n_pairs = min(len(df_strr), len(df_ltrr))

    org_strr = _clean_numeric(df_strr[org_col]).iloc[:n_pairs].reset_index(drop=True)
    new_strr = _clean_numeric(df_strr[new_col]).iloc[:n_pairs].reset_index(drop=True)
    org_ltrr = _clean_numeric(df_ltrr[org_col]).iloc[:n_pairs].reset_index(drop=True)
    new_ltrr = _clean_numeric(df_ltrr[new_col]).iloc[:n_pairs].reset_index(drop=True)

    valid_strr = org_strr.notna() & new_strr.notna()
    valid_ltrr = org_ltrr.notna() & new_ltrr.notna()
    keep = valid_strr & valid_ltrr
    n_valid_pairs = int(keep.sum())

    strr_dropped = df_strr.iloc[:n_pairs].reset_index(drop=True).loc[~valid_strr, [org_col, new_col]]
    ltrr_dropped = df_ltrr.iloc[:n_pairs].reset_index(drop=True).loc[~valid_ltrr, [org_col, new_col]]

    if n_valid_pairs == 0:
        return {
            'analysis_possible': False,
            'message': "No valid paired rows after cleaning.",
            'clean_data': pd.DataFrame(),
            'STRR': None,
            'LTRR': None,
            'summary_table': pd.DataFrame(),
            'figures': {},
            'dropped_rows': {'STRR': strr_dropped, 'LTRR': ltrr_dropped},
            'n_pairs': int(n_pairs),
            'n_valid_pairs': n_valid_pairs,
            'average_org_risk': np.nan,
            'median_org_risk': np.nan,
        }

    org = org_strr[keep].to_numpy()
    new_s = new_strr[keep].to_numpy()
    new_l = new_ltrr[keep].to_numpy()
    red_s = org - new_s
    red_l = org - new_l

    clean_data = pd.DataFrame(
        {
            'Org': org,
            'New_STRR': new_s,
            'New_LTRR': new_l,
            'Red_STRR': red_s,
            'Red_LTRR': red_l,
        }
    )

    STRR = _summarize_strategy(org, new_s, red_s, 'STRR')
    LTRR = _summarize_strategy(org, new_l, red_l, 'LTRR')

    # --- V2-parity: combined/pooled Org risk stats across BOTH strategies ---
    combined_org_risk = pd.concat([
        pd.Series(org_strr[valid_strr]),
        pd.Series(org_ltrr[valid_ltrr])
    ])
    average_org_risk = round(float(combined_org_risk.mean()), 4)
    median_org_risk = round(float(combined_org_risk.median()), 4)

    summary_table = pd.DataFrame(
        [
            {
                'Strategy': 'STRR',
                'N': STRR['n'],
                'Mean Org': STRR['mean_org'],
                'Median Org': STRR['median_org'],
                'Mean New': STRR['mean_new'],
                'Median New': STRR['median_new'],
                'ARR Mean': (STRR['mean_org'] - STRR['mean_new']),
                'ARR Median': (STRR['median_org'] - STRR['median_new']),
                'Std New': STRR['std_new'],
                'Mean ARR': STRR['mean_arr'],
                '95% CI low': STRR['bootstrap_ci_low'],
                '95% CI high': STRR['bootstrap_ci_high'],
                'Sign test p (one-sided)': STRR['sign_test_p_one_sided'],
                'Prop reduced': STRR['prop_reduced'],
                'Prop reduced CI low': STRR['prop_reduced_ci_low'],
                'Prop reduced CI high': STRR['prop_reduced_ci_high'],
                '% Reduced (%)': STRR['percentage_reduced'],
                '% Reduced CI low (%)': STRR['ci_reduced'][0],
                '% Reduced CI high (%)': STRR['ci_reduced'][1],
                '% No Reduction (%)': STRR['percentage_no_reduction'],
                '% No Reduction CI low (%)': STRR['ci_no_reduction'][0],
                '% No Reduction CI high (%)': STRR['ci_no_reduction'][1],
            },
            {
                'Strategy': 'LTRR',
                'N': LTRR['n'],
                'Mean Org': LTRR['mean_org'],
                'Median Org': LTRR['median_org'],
                'Mean New': LTRR['mean_new'],
                'Median New': LTRR['median_new'],
                'ARR Mean': (LTRR['mean_org'] - LTRR['mean_new']),
                'ARR Median': (LTRR['median_org'] - LTRR['median_new']),
                'Std New': LTRR['std_new'],
                'Mean ARR': LTRR['mean_arr'],
                '95% CI low': LTRR['bootstrap_ci_low'],
                '95% CI high': LTRR['bootstrap_ci_high'],
                'Sign test p (one-sided)': LTRR['sign_test_p_one_sided'],
                'Prop reduced': LTRR['prop_reduced'],
                'Prop reduced CI low': LTRR['prop_reduced_ci_low'],
                'Prop reduced CI high': LTRR['prop_reduced_ci_high'],
                '% Reduced (%)': LTRR['percentage_reduced'],
                '% Reduced CI low (%)': LTRR['ci_reduced'][0],
                '% Reduced CI high (%)': LTRR['ci_reduced'][1],
                '% No Reduction (%)': LTRR['percentage_no_reduction'],
                '% No Reduction CI low (%)': LTRR['ci_no_reduction'][0],
                '% No Reduction CI high (%)': LTRR['ci_no_reduction'][1],
            },
        ]
    )

    figs = {}
    if plot:
        plt.style.use("seaborn-v0_8-whitegrid")

        fig, ax = plt.subplots(figsize=(5, 4))

        original = org
        strr = new_s
        ltrr = new_l
        data = [original, strr, ltrr]
        labels = ['Original', 'STRR', 'LTRR']

        box = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            widths=0.5,
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='0.6',
                            markersize=3, alpha=0.4, markeredgecolor='none'),
        )

        for patch in box['boxes']:
            patch.set(facecolor='#e0e0e0', edgecolor='#4d4d4d', alpha=0.8)
        plt.setp(box['whiskers'], color='#4d4d4d', linewidth=1.2)
        plt.setp(box['caps'],     color='#4d4d4d', linewidth=1.2)
        plt.setp(box['medians'],  color='#2b2b2b', linewidth=1.6)

        def _simple_mean_ci(x):
            x = np.asarray(x, dtype=float)
            m = x.mean()
            s = x.std(ddof=1)
            n = len(x)
            ci = 1.96 * s / np.sqrt(n)
            return m, ci

        means, cis = [], []
        for arr in data:
            m, ci = _simple_mean_ci(arr)
            means.append(m)
            cis.append(ci)

        x_pos = np.array([1, 2, 3])

        ax.plot(
            x_pos,
            means,
            color='red',
            marker='o',
            linewidth=2,
            markersize=5,
            label='Mean'
        )

        for x, m, ci in zip(x_pos, means, cis):
            ax.errorbar(
                x=x,
                y=m,
                yerr=ci,
                fmt='none',
                ecolor='#222222',
                elinewidth=1.8,
                capsize=10,
                capthick=1.8,
                alpha=0.9
            )

        ax.set_title("Risks Reduction Comparison", fontsize=11)
        ax.set_ylabel("Predicted Risk", fontsize=10)
        ax.set_xlabel("")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', frameon=False, fontsize=9)

        plt.tight_layout()
        plt.savefig('Risk Comparison box plots_V4.png', dpi=300, bbox_inches='tight')

        figs['box_with_mean'] = fig

    return {
        'analysis_possible': True,
        'clean_data': clean_data,
        'STRR': STRR,
        'LTRR': LTRR,
        'summary_table': summary_table,
        'figures': figs,
        'dropped_rows': {'STRR': strr_dropped, 'LTRR': ltrr_dropped},
        'n_pairs': int(n_pairs),
        'n_valid_pairs': n_valid_pairs,
        'average_org_risk': average_org_risk,
        'median_org_risk': median_org_risk,
    }



def analyze_risk_significance(df, mode, n_resamples=10000, random_state=42):
    orig_raw = pd.to_numeric(df['Org Risk (RB)'], errors='coerce')
    new_raw = pd.to_numeric(df['New risk(RB)'], errors='coerce')
    keep = orig_raw.notna() & new_raw.notna()

    orig = orig_raw[keep].reset_index(drop=True)
    new = new_raw[keep].reset_index(drop=True)

    # Pairwise differences (computed once, after joint alignment)
    diffs = (orig - new).to_numpy()

    # 1. Absolute Risk Reduction (ARR)
    arr = float(np.mean(diffs))

    # 2. Bootstrap 95% CI for the Mean (BCa on precomputed diffs,
    #    with random_state and a small-sample guard)
    if len(diffs) < 2:
        ci_low, ci_high = np.nan, np.nan
    else:
        boot_res = bootstrap(
            (diffs,),
            np.mean,
            confidence_level=0.95,
            n_resamples=n_resamples,
            method='BCa',
            random_state=random_state
        )
        ci_low = float(boot_res.confidence_interval.low)
        ci_high = float(boot_res.confidence_interval.high)

    # 3. Nonparametric p-value (Sign Test) 
    _, p_value_sign = sign_test(diffs, mu0=0)

    # Visualization: Distribution of Differences 
    plt.figure(figsize=(8, 5))
    sns.histplot(diffs, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(arr, color='red', linestyle='--', label=f'Mean ARR: {arr:.4f}')
    plt.axvline(0, color='black', linewidth=1.5)
    plt.title(f'Distribution of Paired Risk Reduction on {mode} Strategy (Org - New)')
    plt.xlabel('Risk Reduction')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    return {
        'Absolute Risk Reduction (ARR)': arr,
        '95% CI Lower': ci_low,
        '95% CI Upper': ci_high,
        'Sign Test p-value': p_value_sign,
        'Significant': ci_low > 0 if ci_low == ci_low else False
    }
    

def plot_recommendations(patient_ids, original_data, recommendation_df, patients_risk_df, features,mode,Health_Guiedance,healthy_ranges=1,CI=0.95, IQR=1,TI=1, rec_indicator=1 ,pp_box=1):
    """
    Function to generate feature plots and boxplots for patient recommendations.

    Parameters:
        patient_ids (list): List of patient IDs.
        original_data (DataFrame): Original dataset containing feature values.
        recommendation_df (DataFrame): DataFrame with recommendations for patients.
        patients_risk_df (DataFrame): DataFrame with risk data for patients.
        features (list): List of features to plot.
        CI (float or None): Confidence interval percentage (e.g., 0.95 for 95%). If None, no CI bar is plotted.
        IQR (int): 1 to plot IQR box for population distribution, 0 to skip.
        pp_box (int): 1 to include population boxplot in risk evaluation, 0 to skip.

    Returns:
        None: Displays the plots.
    """
    # Marker size parameters
    marker_size = 5  # Size of markers from 10 to 5
    marker_edge_width = 1.5  # Thickness of marker edges

    num_features = len(features)
    num_cols = 4
    num_rows = -(-num_features // num_cols) + 1  # Add one row for boxplots

    fig = plt.figure(figsize=(3.5 * num_cols, 3 * num_rows)) # change from 5 to 3
    fig.suptitle(f'The Recommendations ({mode})',weight='bold', fontsize=20)

    # Add feature plots
    for i, feature in enumerate(features):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)

        # Plot population CI if CI is not None
        if CI:
            feature_data = original_data[feature]
            # ci_lower, ci_upper = calculate_ci(feature_data, ci=CI)
            alpha = 1-CI
            ci_lower = feature_data.quantile(alpha/2)
            ci_upper = feature_data.quantile(1-alpha/2)
            ax.bar('CI', ci_upper - ci_lower, bottom=ci_lower, color='g', alpha=0.5)


        # Plot IQR if enabled
        if IQR:
            q25 = original_data[feature].quantile(0.25)
            q50 = original_data[feature].quantile(0.50)  # Median line
            q75 = original_data[feature].quantile(0.75)
            if TI:
                feature_data = original_data[feature]
                ti_lower, ti_upper = calculate_tolerance_interval(feature_data, confidence=0.95, proportion=0.95)

                # Plot TI as a light green bar
                ax.bar(feature, ti_upper - ti_lower, bottom=ti_lower, color='lightgreen', alpha=0.5)
                ax.bar(feature, q75 - q25, bottom=q25, color='lightblue', alpha=0.7)
                # Add dashed horizontal lines for IQR thresholds
                
                # Add median line for IQR
                ax.hlines(q50, -0.4, 0.4, colors='orange', linewidth=2)
            else:

                q25 = original_data[feature].quantile(0.25)
                q50 = original_data[feature].quantile(0.50)  # Median line
                q75 = original_data[feature].quantile(0.75)

                # Plot IQR as a light blue bar
                ax.bar(feature, q75 - q25, bottom=q25, color='lightblue', alpha=0.7)
                # Add median line for IQR
                ax.hlines(q50, -0.4, 0.4, colors='orange', linewidth=2)
            
        
        if healthy_ranges:
            lower_h = Health_Guiedance[Health_Guiedance['Feature']==feature].iloc[0,1]
            upper_h = Health_Guiedance[Health_Guiedance['Feature']==feature].iloc[0,2]
            
            # Plot finite healthy bounds
            if pd.notnull(lower_h) and not np.isinf(lower_h):
                ax.axhline(y=lower_h, color='r', linestyle='--', linewidth=1)
            elif lower_h == float('-inf') or lower_h == -np.inf:
                ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
                
            if pd.notnull(upper_h) and not np.isinf(upper_h):
                ax.axhline(y=upper_h, color='g', linestyle='--', linewidth=1)
            
        # Plot patient intervals and current/recommended values
        for j, patient_id in enumerate(patient_ids):
            patient_recommendation = recommendation_df[(recommendation_df['Patient_IDs'] == patient_id) &(recommendation_df['Feature'] == feature)]

            # Adjust the x-coordinate based on whether CI is included
            x_coord = j + 1 #if CI is not None else j + 1

            if not patient_recommendation.empty:
                current_val = patient_recommendation['Current'].values[0]
                recommended_val = patient_recommendation['Rounded Recommended'].values[0]
                min_val = patient_recommendation['Min'].values[0]
                max_val = patient_recommendation['Max'].values[0]
                # if patient_recommendation['Min'].values[0] is :

                # Plot interval bar
                ax.bar(x_coord, max_val - min_val,
                       bottom=min_val,
                       color='blue', alpha=0.5)

                # Plot current value as red x above green circle
                ax.plot(x_coord, current_val + 0.02,
                        'rx', markersize=marker_size + 3, markeredgewidth=marker_edge_width)
                if rec_indicator:
                    # Plot recommended value as green circle below red x
                    ax.plot(x_coord, recommended_val,'go', markersize=marker_size)

            else:
                current_val = original_data.loc[patient_id, feature]
                ax.plot(x_coord, current_val + 0.02,
                        'rx', markersize=marker_size + 3, markeredgewidth=marker_edge_width)
                if rec_indicator:
                
                    ax.plot(x_coord, current_val,'go', markersize=marker_size) 
                    
            # Add grid to the plot
            ax.grid(True, linestyle='-', alpha=0.7)
                
            

        # Set titles and labels
        ax.set_title(feature,fontsize=10, pad=4) # Move the feature name down a bit
        ax.set_xlabel('') #Patients

        # Focused Adaptive Y-axis Scaling
        y_vals = []
        
        # 1. Population summary elements (CI, TI, IQR)
        if CI: y_vals.extend([ci_lower, ci_upper])
        if 'q25' in locals(): y_vals.extend([q25, q75]) 
        if TI and 'ti_lower' in locals(): y_vals.extend([ti_lower, ti_upper])
        
        # 2. Healthy Guidance (only finite values)
        if healthy_ranges:
            if pd.notnull(lower_h) and not np.isinf(lower_h): y_vals.append(lower_h)
            if pd.notnull(upper_h) and not np.isinf(upper_h): y_vals.append(upper_h)
            
        # 3. Patient Recommendations for all selected patients
        selected_recs = recommendation_df[(recommendation_df['Feature'] == feature) & (recommendation_df['Patient_IDs'].isin(patient_ids))]
        if not selected_recs.empty:
            # Include all plotted markers and intervals
            y_vals.extend(selected_recs[['Min', 'Max', 'Current', 'Rounded Recommended']].values.flatten())
            
        # Filter for finite values only
        y_vals = [v for v in y_vals if pd.notnull(v) and not np.isinf(v)]
        
        if y_vals:
            y_min_f, y_max_f = min(y_vals), max(y_vals)
            y_range = y_max_f - y_min_f
            buffer = max(y_range * 0.1, 0.1) # 10% margin, at least 0.1
            ax.set_ylim(y_min_f - buffer, y_max_f + buffer)
        
        # Adjust x-ticks and labels based on whether CI is included
        if CI is not None and TI == 1:
            ax.set_xticks(range(len(patient_ids) + 3))  # +3 for 'CI', 'TI', and 'IQR'
            ax.set_xticklabels(['CI', 'TI', 'IQR'] + [f'X{j+1}' for j in range(len(patient_ids))], fontsize=8)

        elif TI == 1:
            ax.set_xticks(range(len(patient_ids) + 2))  # +2 for 'TI' and 'IQR'
            ax.set_xticklabels(['TI', 'IQR'] + [f'X{j+1}' for j in range(len(patient_ids))], fontsize=8)
        else:
            ax.set_xticks(range(len(patient_ids) + 1))  # +1 for 'CI'
            ax.set_xticklabels(['CI'] + [f'X{j+1}' for j in range(len(patient_ids))], fontsize=8)
        
        # Adjust x-axis limits (no changes needed here)
        ax.set_xlim(-0.5, len(patient_ids) + 0.5) #if CI is not None else len(patient_ids) + 0.5)


            # Add consolidated boxplot for risks spanning two columns if pp_box is enabled
    if pp_box:
        ax_risk = fig.add_subplot(num_rows, num_cols, (num_features + 1, num_features + 2))

        q25r = patients_risk_df['Org Risk (RB)'].quantile(0.25)
        q50r = patients_risk_df['Org Risk (RB)'].quantile(0.50)
        q75r = patients_risk_df['Org Risk (RB)'].quantile(0.75)

    # Add median line for IQR

        ax_risk.axhline(y=q25r, xmin=0, xmax=1, color='gray', linestyle='--')
        ax_risk.axhline(y=q50r, xmin=0, xmax=1, color='orange', linewidth = 1)
        ax_risk.axhline(y=q75r, xmin=0, xmax=1, color='gray', linestyle='--')


        
        
        risks_per_patient = []
        for j in range(len(patient_ids)):
            risks_per_patient.append(patients_risk_df['Org Risk (RB)'])
        
        # Add indicators for each patient's risks
        for j, patient_id in enumerate(patient_ids):
            patient_risk = patients_risk_df[patients_risk_df['Patient_ID'] == patient_id]

            if not patient_risk.empty:
                current_risk = patient_risk['Org Risk (RB)'].values[0]
                new_risk = patient_risk['New risk(RB)'].values[0]

                # Plot current risk as red x above green circle
                ax_risk.plot(j + 1,
                             current_risk ,
                             'rx', markersize=marker_size + 3, markeredgewidth=marker_edge_width) #current_risk + 0.02

                # Plot new risk as green circle below red x
                ax_risk.plot(j + 1,
                             new_risk,
                             'go', markersize=marker_size)

                selected_patients_risk_df = patients_risk_df[patients_risk_df['Patient_ID'].isin(patient_ids)]
                all_risks = []
                all_risks.extend(selected_patients_risk_df['Org Risk (RB)'].tolist())
                all_risks.extend(selected_patients_risk_df['New risk(RB)'].tolist())
                
                

                # Adaptive Y-axis for ax_risk
                y_min_risk = min(all_risks)
                y_max_risk = max(all_risks)        
                # Add a buffer of 10% to the top and bottom
                buffer_risk = 0.1 * (y_max_risk - y_min_risk)
                y_min_risk -= buffer_risk
                y_max_risk += buffer_risk
        
                     # Set y-axis ticks
                yticks = np.arange(math.floor(y_min_risk*10)/10, math.ceil(y_max_risk*10)/10, 0.1) #ensure that value will looks nice
                ax_risk.set_yticks(yticks)
        
                ax_risk.set_ylim(y_min_risk, y_max_risk)

        
        
        fig.supylabel('Feature Values', fontsize=14, fontweight='bold',x=0.06, y=0.6) # global y-axsis
          #Add vertical line to the left of y-axis label
            #             ax.set_ylabel('Feature Values', fontsize=12, fontweight='bold')

        # Get the position of the current subplot
        bbox = ax.get_position()

        # Calculate the line position and length
        line_x = bbox.x0 - 0.245  # Adjust this value to move the line left or right
        line_ymin = bbox.y0 - 0.13 * bbox.height  # Adjust this value to change the bottom of the line
        line_ymax = bbox.y1 + 2.8 * bbox.height  # Adjust this value to change the top of the line

        line = Line2D([line_x, line_x], [line_ymin, line_ymax], 
                      transform=fig.transFigure, figure=fig, 
                      color='black', linewidth=1.8)
        fig.lines.append(line)

        # Set titles and labels for the consolidated boxplot
        ax_risk.set_title('Patient Risk Indicators',fontsize=10, pad=4)
        # ax_risk.set_xlabel('Patients')
        ax_risk.set_ylabel('Risk Values',labelpad=-3)
        ax_risk.set_xticks(range(1, len(patient_ids) + 1))
        ax_risk.set_xticklabels([f'X{j+1}' for j in range(len(patient_ids))], fontsize=8)
    
        # Add grid to the risk indicators plot
        ax_risk.grid(True, linestyle='-', alpha=0.7, color='gray', linewidth=0.5)
        

        # Add horizontal line above x-axis label
        line = Line2D([0.083, 0.9],  #horizontal_line_left, horizontal_line_right
                      [0.29, 0.29], 
                      transform=fig.transFigure, 
                      figure=fig, 
                      color='black', 
                      linewidth=1.8)
        fig.lines.append(line)


        fig.supxlabel('Patients',y=0.26, fontsize=14, fontweight='bold')

    

    # Add legend in the top-right corner with custom entries
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.5, label='95% CI') ,            # Gray box for IQR is mpatches.Patch(color='lightgray', alpha=0.5, label='IQR (50% of data)')
        mpatches.Patch(color='lightgreen', alpha=0.5, label='95% Tolerance interval') if TI else Line2D([0], [0], color='g', linestyle='--', linewidth=1, label='Healthy Upper Limit')  , # Dashed gray lines for thresholds
        mpatches.Patch(color='blue', alpha=0.3, label='Recommended Region'),                # Blue bar for recommended region
        Line2D([0], [0], color='r', linestyle='--', linewidth=1, label='Healthy Lower Limit'),
        Line2D([0], [0], color='red', marker='x', markersize=10, markeredgewidth=2,
               linestyle='', label='Current Value'),        # Red 'x' for current value
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='IQR'),
    ]


    # Add legend outside the plot in the top-right corner with three columns
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(2.3, 3.85),  # Centered above the plot
        ncol=3,  # Split into three columns
        frameon=True,  # Add a box around the legend
        fontsize=10
    )

    plt.savefig(f'Recommendation Figure {mode}', bbox_inches='tight')
    plt.show()  
    
