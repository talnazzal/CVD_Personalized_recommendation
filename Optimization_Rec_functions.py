import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_categorical_dtype
)
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

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.stats import spearmanr
import xmlrpc.client
import time
import re
from gams import GamsWorkspace #pip install gamsapi
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

##### Preprocessing Functions########
#Helpful feature sets
changable_feature = ['ETHANL03', 'CURSMK01', 'MOVE', 'P_CARB', 'P_PROT', 'P_SFAT', 'P_TFAT','BMI01', 'CHOL', 'DFIB', 'TOTCAL03']
ind_changable = ['HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31', 'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09']
feature_set = changable_feature + ind_changable


changable_feature_ordinal = ['ETHANL03_ordinal', 'CURSMK01_ordinal', 'MOVE_ordinal','P_CARB_ordinal', 'P_PROT_ordinal', 'P_SFAT_ordinal', 'P_TFAT_ordinal','BMI01_ordinal', 'CHOL_ordinal', 'DFIB_ordinal', 'TOTCAL03_ordinal']
ind_changable_ordinal = ['HDLSIU02_ordinal', 'LDLSIU02_ordinal', 'TCHSIU01_ordinal', 'TRGSIU01_ordinal','SBPA21_ordinal', 'SBPA22_ordinal', 'ANTA07A_ordinal','ANTA07B_ordinal', 'ECGMA31_ordinal', 'HMTA03_ordinal',
       'APASIU01_ordinal', 'APBSIU01_ordinal', 'LIPA08_ordinal','CHMA09_ordinal']
unchangeable= [ 'CIGTYR01', 'ELEVEL01', 'GENDER','RACEGRP', 'V1AGE01', 'DIABTS', 'HYPERT04', 'HYPTMDCODE01', 'CHOLMDCODE01','ANTA01', 'CIGT01', 'HYPTMD01', 'ANTICOAGCODE01', 'ASPIRINCODE01', 'STATINCODE01']

feature_set_ordinal = changable_feature_ordinal + ind_changable_ordinal

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

    # Impute continuous variables with median
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
    
    for var in continuous_vars:
        if var in df.columns:
            # Create bin edges based on quantiles
            bin_edges = pd.qcut(df[var], q=n_bins, duplicates='drop', retbins=True)[1]
            
            # Discretize the variable
            df_discretized[f"{var}_ordinal"] = pd.cut(df[var], bins=bin_edges, labels=False, include_lowest=True)
            
            # Add 1 to shift the range from 0-4 to 1-5
            df_discretized[f"{var}_ordinal"] += 1
            
            # Drop the original continuous variable
            df_discretized = df_discretized.drop(columns=[var])
        else:
            print(f"Warning: Variable '{var}' not found in the dataset.")
    
    return df_discretized


##### Evaluation ####
def evaluate_and_visualize_model(model, X_test, y_test):
    """
    Evaluate model performance with ROC, PR curves, confusion matrices,
    and show the best F1 threshold on the PR plot.
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate confusion matrix (default threshold=0.5)
    cm_default = confusion_matrix(y_test, y_pred)
    
    # Print default threshold confusion matrix
    print("\nConfusion Matrix (Default Threshold = 0.5):")
    print("                 Predicted")
    print("                 Class 0  Class 1")
    print(f"Actual Class 0:  {cm_default[0,0]:6d}  {cm_default[0,1]:6d}")
    print(f"Actual Class 1:  {cm_default[1,0]:6d}  {cm_default[1,1]:6d}")

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
    
    # Print optimal threshold confusion matrix
    print(f"\nConfusion Matrix (Optimal Threshold = {optimal_threshold:.3f}):")
    print("                 Predicted")
    print("                 Class 0  Class 1")
    print(f"Actual Class 0:  {cm_optimal[0,0]:6d}  {cm_optimal[0,1]:6d}")
    print(f"Actual Class 1:  {cm_optimal[1,0]:6d}  {cm_optimal[1,1]:6d}")

    # Create ROC and PR curves plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', color='blue')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_title('ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.grid(True)

    # PR Curve
    pr_auc = average_precision_score(y_test, y_pred_proba)
    ax2.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})', color='green')
    
    # Plot the point for the best F1 threshold
    ax2.scatter(recall[optimal_idx], precision[optimal_idx],
                color='red',
                label=(f'Optimal F1 = {best_f1_score:.3f}\n'
                       f'Threshold = {optimal_threshold:.3f}'))
    
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



##### Optimization Functions ######
def check_rule(df, rule, sample_idx):
    # Get the values of columns for the given sample index
    sample = df.loc[sample_idx]
    
    # Remove parentheses and split the rule by 'and' or '&' to get individual conditions
    if ' and ' in rule:
        conditions = rule.replace('(', '').replace(')', '').split(' and ')
    else:
        conditions = rule.split(' & ')
    
    # Evaluate each condition
    for condition in conditions:
        # Split the condition by spaces to get variable, operator, and value
        parts = condition.strip().split()
        var = parts[0]
        op = parts[1]
        val = float(parts[2])
        
        # Evaluate the condition based on the comparison operator
        if op == '>':
            if not sample[var] > val:
                return 0
        elif op == '>=':
            if not sample[var] >= val:
                return 0
        elif op == '<':
            if not sample[var] < val:
                return 0
        elif op == '<=':
            if not sample[var] <= val:
                return 0
        else:
            raise ValueError(f"Invalid comparison operator: {op}")
    
    # If all conditions are met, return 1
    return 1
def count_feature_occurrences(rules, changeable_features):
    feature_counts = {feature: 0 for feature in changeable_features}
    
    for rule in rules:
        for feature in changeable_features:
            # Check for exact match or various suffix matches
            if (feature in rule or 
                f"{feature}_suffix" in rule or 
                f"{feature}_ordinal" in rule or
                rule.startswith(feature)):
                feature_counts[feature] += 1
    
    return feature_counts
def embed_rules_vectorized(df, rules_df, outcome_column=None, rule_col='rule', keep_original=False):
    df_copy = df.copy()
    
    if not isinstance(rules_df, pd.DataFrame):
        raise TypeError("'rules_df' must be a pandas DataFrame.")
    
    if rule_col not in rules_df.columns:
        raise ValueError(f"DataFrame 'rules_df' must contain a {rule_col} column.")
    
    rules = rules_df[rule_col].tolist()
    
    rule_results = {}
    
    for rule in rules:
        col_name = rule  # Use the rule itself as the column name
        try:
            rule_results[col_name] = df_copy.eval(rule).astype(int)
        except Exception as e:
            raise ValueError(f"Error processing rule '{rule}': {e}")
    
    rules_df = pd.DataFrame(rule_results, index=df_copy.index)
    
    if keep_original:
        result_df = pd.concat([df_copy, rules_df], axis=1)
    else:
        result_df = rules_df
    
    # Check if outcome_column exists and move it to the end if it does
    if outcome_column and outcome_column in df_copy.columns:
        if outcome_column not in result_df.columns:
            result_df[outcome_column] = df_copy[outcome_column]
        result_df = result_df[[col for col in result_df.columns if col != outcome_column] + [outcome_column]]
    
    return result_df

def calculate_Qj_Uj(patient_id, Criterias, X_test, mode, indirectly_changable_features, unchangeable_features):
    # Define fixed features based on mode
    if mode == 'LTRR':
        fixed_features = unchangeable_features 
    elif mode == 'STRR' :
        fixed_features = unchangeable_features + indirectly_changable_features

    # Ensure Patient_ID is in the index of X_test
    if patient_id not in X_test.index:
        return "Patient ID not found in dataset"

    # Extract patient's data
    patient_data = X_test.loc[patient_id]

    Qj = []
    Uj = []
    for criterion in Criterias:
        # Split criterion into feature, operator, and value
        feature, operator, value = criterion.split(' ')
        value = float(value)

        # Check if the feature is in the fixed features
        if feature not in fixed_features:
            Qj.append(0)
            Uj.append(0)
        else:
            # Evaluate if the criterion is satisfied or not
            if operator == '>' and patient_data[feature] > value:
                Qj.append(0)
                Uj.append(1)
            elif operator == '<=' and patient_data[feature] <= value:
                Qj.append(0)
                Uj.append(1)
            else:
                Qj.append(1)
                Uj.append(0)

    return Qj, Uj
    
def calculate_C(rule_df, col_name = 'rule'):
    C = []
    for rule in rule_df[col_name]:
        # Count the number of criteria in the rule
        # criteria in a rule are separated by 'and'
        num_criteria = len(rule.split('and'))
        C.append(num_criteria)
    return C
    
def calculate_P(rule_df,col_name = 'rule'):
    max_criteria_count = 0
    for rule in rule_df[col_name]:
        # Split rule into criteria and count them
        criteria_count = len(rule.split('and'))
        # Update max_criteria_count if current count is higher
        if criteria_count > max_criteria_count:
            max_criteria_count = criteria_count
    return max_criteria_count
    
def calculate_A(rule_df, Criterias, col_name = 'rule'):
    num_rules = len(rule_df)
    num_criterias = len(Criterias)

    # Initialize A matrix with zeros
    A = np.zeros((num_rules, num_criterias))

    for i, rule in enumerate(rule_df[col_name]):
        for j, criterion in enumerate(Criterias):
            # Check if the criterion is part of the rule
            if criterion in rule:
                A[i, j] = 1
    return A

def calculate_Nj(patient_id, Criterias, X_test):
    # Define fixed features based on mode
    fixed_features = X_test.columns
    # Ensure Patient_ID is in the index of X_test
    if patient_id not in X_test.index:
        return "Patient ID not found in dataset"
    Nj = []
    for criterion in Criterias:
        Nj.append(check_rule(X_test,criterion,patient_id))
        # Split criterion into feature, operator, and value
    return Nj
def calculate_F(rule_df, Criterias,D, col_name = 'rule'):
    num_rules = len(rule_df)
    num_criterias = len(Criterias)

    # Initialize A matrix with zeros
    F = np.zeros((num_rules, num_criterias))

    for i, rule in enumerate(rule_df[col_name]):
        for j, criterion in enumerate(Criterias):
            # Check if the criterion is part of the rule
            if criterion in rule:
                F[i, j] = 1
                inclusion_criteria = list(np.where(D[j]==1)[0])
                F[i, inclusion_criteria] = 1
            
    return F

def check_criteria_V(criterias):
    n = len(criterias)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c1 = criterias[i]
            c2 = criterias[j]
            c1_var, c1_op, c1_val = c1.split()
            c2_var, c2_op, c2_val = c2.split()
            
            if c1_var != c2_var or c1_op != c2_op:
                continue
            
            c1_val, c2_val = float(c1_val), float(c2_val)
            
            if c1_op in ('>', '>='):
                if c1_val <= c2_val:
                    matrix[i, j] = 1
            elif c1_op in ('<', '<='):
                if c1_val >= c2_val:
                    matrix[i, j] = 1
    
    return matrix

def check_criteria_D(criterias):
    n = len(criterias)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c1 = criterias[i]
            c2 = criterias[j]
            c1_var, c1_op, c1_val = c1.split()
            c2_var, c2_op, c2_val = c2.split()
            
            if c1_var != c2_var or c1_op != c2_op:
                continue
            
            c1_val, c2_val = float(c1_val), float(c2_val)
            
            if c1_op in ('>', '>='):
                if c1_val >= c2_val:
                    matrix[i, j] = 1
            elif c1_op in ('<', '<='):
                if c1_val <= c2_val:
                    matrix[i, j] = 1
    
    return matrix
    
def check_criteria_G(criterias):
    n = len(criterias)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            c1 = criterias[i]
            c2 = criterias[j]
            c1_var, c1_op, c1_val = c1.split()
            c2_var, c2_op, c2_val = c2.split()
            
            if c1_var != c2_var:
                continue
            
            c1_val, c2_val = float(c1_val), float(c2_val)
            
            exclusive_pairs = [('>=', '<'), ('>', '<='), ('<=', '>'), ('<', '>=')]
            
            if (c1_op, c2_op) in exclusive_pairs and c1_val == c2_val:
                matrix[i, j] = 1
            elif (c1_op in ('>', '>=') and c2_op in ('<', '<=') and c2_val > c1_val):
                matrix[i, j] = 1
            elif (c1_op in ('<', '<=') and c2_op in ('>', '>=') and c1_val > c2_val):
                matrix[i, j] = 1
    
    return matrix
def check_criteria_H(criterias):
    n = len(criterias)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            c1 = criterias[i]
            c2 = criterias[j]
            c1_var, c1_op, c1_val = c1.split()
            c2_var, c2_op, c2_val = c2.split()
            
            if c1_var != c2_var:
                continue
            
            c1_val, c2_val = float(c1_val), float(c2_val)
            
            exclusive_pairs = [('>=', '<'), ('>', '<='), ('<=', '>'), ('<', '>=')]
            
            if (c1_op, c2_op) in exclusive_pairs and c1_val == c2_val:
                matrix[i, j] = 1
            elif (c1_op in ('>', '>=') and c2_op in ('<', '<=') and c2_val < c1_val):
                matrix[i, j] = 1
            elif (c1_op in ('<', '<=') and c2_op in ('>', '>=') and c2_val > c1_val):
                matrix[i, j] = 1
    
    return matrix


def generate_S(A):
    """
    Generates the S(i,j) matrix where S[i][j] = number of criteria in rule i excluding j.
    
    Args:
        A (np.ndarray): Binary matrix of shape (n_rules, n_criteria) where A[i][j] = 1 
                        indicates rule i uses criteria j.
    
    Returns:
        np.ndarray: S matrix of shape (n_rules, n_criteria), where S[i][j] is defined 
                    only for A[i][j] = 1 and equals the count of other criteria in rule i.
    """
    # Calculate total criteria per rule (sum over columns)
    criteria_per_rule = np.sum(A, axis=1)
    
    # Compute S(i,j) = total criteria in rule i - 1 (exclude current j)
    S = (criteria_per_rule[:, np.newaxis] - 1) * A
    
    return S.astype(int)  # Ensure integer type


#Convert a list to GAMS parameter format
def list_to_gams_param(lst):
    gams_param = "\n".join(f"{i+1} {val}" for i, val in enumerate(lst))
    gams_param = "/\n{}\n/".format(gams_param)
    return gams_param
    
#Convert a 2D numpy array to a GAMS table with row and column names
def matrix_to_gams_table(matrix):
    # create a list of tuples with the row and column indices and the matrix values
    gams_set = [(i+1, value) for i, row in enumerate(matrix) for value in row ]
    # create a string representation of the GAMS table with the row and column names
    row_names = [f"{i+1}" for i in range(matrix.shape[0])]
    col_names = [f"{j+1}" for j in range(matrix.shape[1])]
    col_values = ["\t".join(str(val) for val in row) for row in matrix]
    gams_param = "\n".join(f"{row_name}\t{col_value}" for row_name,col_value in zip(row_names,col_values))
    gams_param = "\t{}\n{}".format("\t".join(col_names), gams_param)
    return gams_param

def Parameters(rules_df):
    ## Global parameters
    Criterias = extract_criterias(rules_df)
    B = rules_df.coef.tolist()
    C = calculate_C(rules_df) # C_i is the number of criterion in rule i
    Pj_dic ={} 
    for criterion in Criterias:
        criterion_count = 0
        for rule in rules_df.rule:
            if criterion in rule:
                criterion_count+= 1
        if criterion_count == 0:
            print(f'criteria {criterion} not found in any rule')
            break
        Pj_dic[criterion] = criterion_count

    P=list(Pj_dic.values()) #P_j is the number of times criteria j is repeated among all rules
    A = calculate_A(rules_df, Criterias) # represents whether criteria 𝑗 is used in the rule 𝑖 {1: yes, 0: No}.
    
    
    V = check_criteria_V(Criterias) #represent if the 𝑗 criteria is not satisfied then the 𝑘 criteria should also not be satisfied {1: True, 0: False}.
    
    D = check_criteria_D(Criterias)
    F = calculate_F(rules_df, Criterias, D)
    S= generate_S(A)

    G = check_criteria_G(Criterias) #represent if the 𝑗 criteria is not satisfied then the 𝑘 criteria should be satisfied {1: True, 0: False}.
    H = check_criteria_H(Criterias) #represent if the 𝑗 criteria is satisfied then the 𝑘 criteria should not be satisfied {1: True, 0: False}.
    return Criterias,B,C,P,A,F,V,D,G,H,S
def extract_neos_values(neos_output):
    # Check for infeasibility
    if "Model Status      10 Integer Infeasible" in neos_output or "No solution returned" in neos_output:
        print("The GAMS model is infeasible. No feasible solution was found.")
        return [], [], None  # Empty lists for z_values and x_values, and None for o_value

    # Regular expression pattern for matching indices
    index_pattern = r'(\d+)\s+1\.000'

    # Extracting z_values and x_values
    try:
        z_section = re.search(r'VARIABLE z\.L\s+([\s\S]+?)\n\n\n----', neos_output).group(1)
        x_section = re.search(r'VARIABLE x\.L\s+([\s\S]+?)\n\n\n----', neos_output).group(1)
    except AttributeError:
        # Pattern not found in the output, indicating potential issues
        print("Unable to find VARIABLE sections in NEOS output.")
        return [], [], None

    z_values = re.findall(index_pattern, z_section)
    x_values = re.findall(index_pattern, x_section)

    # Convert strings to integers
    z_values = [int(i) for i in z_values]
    x_values = [int(i) for i in x_values]

    # Extracting o value
    o_value_match = re.search(r'VARIABLE o\.L\s+=\s+([-\d\.]+)', neos_output)
    o_value = float(o_value_match.group(1)) if o_value_match else None

    return z_values, x_values, o_value


def save_dataframe_to_excel(df, file_path, max_attempts=10, wait_seconds=10):
    """
    Tries to save a DataFrame to an Excel file, with retries on PermissionError.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        file_path (str): The path to the Excel file.
        max_attempts (int): Maximum number of attempts to save the file.
        wait_seconds (int): Number of seconds to wait between attempts.
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            df.to_excel(file_path, index=False, engine='openpyxl')
            print(f"File saved successfully to {file_path}")
            break
        except PermissionError:
            attempt += 1
            print(f"Attempt {attempt} failed. Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)
    else:
        print(f"Failed to save file after {max_attempts} attempts.")

def validate_patient_features(data, patient_id, opt_criteria, rules_df, Criterias, model_intercept, coef_column = 'coef'):
    # Get patient data
    patient_data = data.loc[patient_id]

    if len(opt_criteria) == 0:
        # Calculate original risk
        org_embedded = embed_rules_vectorized(patient_data.to_frame().T, rules_df)
        org_risk = 1 / (1 + np.exp(-(np.dot(org_embedded.values, rules_df[coef_column].values) + model_intercept)))
    
        # For each feature, current value = min = max = recommended (no change)
        no_change_rows = []
        for feature in feature_set_ordinal:
            current = patient_data[feature]
            no_change_rows.append({
                'Feature': feature,
                'Min': current,
                'Max': current,
                'Current': current,
                'Recommended': current,
                'Rounded Recommended': current,
                'Difference': 0,
                'Org Risk': org_risk[0],
                'Partial Risk': org_risk[0],   # no modification, risk unchanged
                'Risk Decreased': 'No'
            })
        return pd.DataFrame(no_change_rows)
    # Process rules into feature ranges
    feature_ranges = process_decision_rules(opt_criteria, Criterias, feature_set) #chose version 2 

    # Prepare results storage
    results = []
    
    # Calculate original risk
    org_embedded = embed_rules_vectorized(patient_data.to_frame().T, rules_df)
    org_risk = 1 / (1 + np.exp(-(np.dot(org_embedded.values, rules_df[coef_column].values) + model_intercept)))

    for _, feature_row in feature_ranges.iterrows():
        feature = feature_row['Feature']
        f_min = feature_row['Min']
        f_max = feature_row['Max']
        current = patient_data[feature]
        
        # Check if value needs adjustment
        needs_adjustment = False
        if (f_min is not None and current < f_min) or (f_max is not None and current > f_max):
            needs_adjustment = True
            
        if not needs_adjustment:
            continue
            
        # Calculate recommended value
        recommended = current
        if f_min is not None and current < f_min:
            recommended = f_min
        elif f_max is not None and current > f_max:
            recommended = f_max
            
        # Round recommended value
        if recommended < current:
            rounded_rec = np.floor(recommended)
        else:
            rounded_rec = np.ceil(recommended)
            
        # Calculate partial risk
        modified_data = patient_data.copy()
        modified_data[feature] = rounded_rec
        partial_embedded = embed_rules_vectorized(modified_data.to_frame().T, rules_df)
        partial_risk = 1 / (1 + np.exp(-(np.dot(partial_embedded.values, rules_df[coef_column].values)+ model_intercept)))
        
        # Store results
        results.append({
            'Feature': feature,
            'Min': f_min,
            'Max': f_max,
            'Current': current,
            'Recommended': recommended,
            'Rounded Recommended': rounded_rec,
            'Difference': rounded_rec - current ,
            'Org Risk': org_risk[0],
            'Partial Risk': partial_risk[0],
            'Risk Decreased': 'Yes' if partial_risk < org_risk else 'No'
        })
    
    return pd.DataFrame(results)

    
def process_decision_rules(rules, original_criteria, feature_set):
    # Initialize storage for feature bounds
    features = {}
    
    # 1. Identify non-optimal rules and create inverted constraints
    non_optimal_rules = [rule for rule in original_criteria if rule not in rules]
    inverted_ops = {'>': '<=', '>=': '<', '<': '>=', '<=': '>'}
    
    # Process NON-OPTIMAL rules first to set basic ranges
    for rule in non_optimal_rules:
        try:
            feature, condition = rule.split(' ', 1)
            operator, value = condition.split(' ', 1)
            value = float(value)
            
            # Invert the operator
            inv_operator = inverted_ops[operator]
            
            if feature not in features:
                features[feature] = {
                    'min': float('-inf'),
                    'max': float('inf')
                }
            
            # Update bounds with INVERTED condition
            if inv_operator in ('>', '>='):
                new_min = value if inv_operator == '>' else value
                features[feature]['min'] = max(features[feature]['min'], new_min)
            elif inv_operator in ('<', '<='):
                new_max = value if inv_operator == '<' else value
                features[feature]['max'] = min(features[feature]['max'], new_max)
                
        except Exception as e:
            print(f"Invalid non-optimal rule: '{rule}' - {e}")
            continue

    # 2. Process OPTIMAL rules to tighten ranges
    for rule in rules:
        try:
            feature, condition = rule.split(' ', 1)
            operator, value = condition.split(' ', 1)
            value = float(value)
            
            if feature not in features:
                features[feature] = {
                    'min': float('-inf'),
                    'max': float('inf')
                }
            
            # Update bounds with ORIGINAL condition
            if operator in ('>', '>='):
                new_min = value if operator == '>' else value
                features[feature]['min'] = max(features[feature]['min'], new_min)
            elif operator in ('<', '<='):
                new_max = value if operator == '<' else value
                features[feature]['max'] = min(features[feature]['max'], new_max)
                
        except Exception as e:
            print(f"Invalid rule format: '{rule}' - {e}")
            continue

    # 3. Prepare initial results
    results = []
    for feature, bounds in features.items():
        if bounds['min'] > bounds['max']:
            print(f"Contradiction detected for {feature}: {bounds}")
            continue
            
        results.append({
            'Feature': feature,
            'Min': bounds['min'],
            'Max': bounds['max']
        })
    
    df = pd.DataFrame(results)
    
    # 4. Add missing features from feature_set with suffix handling
    existing_features = df['Feature'].tolist() if not df.empty else []
    
    for base_feature in feature_set:
        # Find all criteria related to this base feature
        related_criteria = [
            c for c in original_criteria 
            if c.split(' ', 1)[0].startswith(base_feature)
        ]
        
        # Find if any full feature name needs to be added
        for criterion in related_criteria:
            full_feature = criterion.split(' ', 1)[0]
            if full_feature in existing_features:
                continue
                
            # Initialize with infinite bounds
            min_val = float('-inf')
            max_val = float('inf')
            
            # Process all related non-optimal criteria
            for c in related_criteria:
                if c in rules:
                    continue  # Skip optimal rules
                    
                try:
                    _, condition = c.split(' ', 1)
                    op, val = condition.split(' ', 1)
                    val = float(val)
                    
                    # Invert the operator
                    inv_op = inverted_ops[op]
                    
                    if inv_op in ('>', '>='):
                        min_val = max(min_val, val if inv_op == '>' else val)
                    elif inv_op in ('<', '<='):
                        max_val = min(max_val, val if inv_op == '<' else val)
                        
                except Exception as e:
                    print(f"Error processing {c}: {e}")
                    continue
            
            # Add to results if not already present
            if full_feature not in existing_features:
                df = pd.concat([df, pd.DataFrame([{
                    'Feature': full_feature,
                    'Min': min_val,
                    'Max': max_val
                }])], ignore_index=True)
                existing_features.append(full_feature)
    
    return df
 




def evaluate_and_apply_recommendations_Lifestyle_ONLY(embedded_X_test,X_test, patient_index, mode, intercept, x_indices, z_indices, A, B, C, Q_indices, U_indices, rules_df, uncontrolable_indices,lifstyle_rule_indices_org, print_results =False, base_0 = True, rule_column= 'rule'): # must identify (lifstyle_rule_indices_org) when apply and mode
    """
    Enhanced version with accurate conflict detection and risk calculation
    
    Key improvements:
    1. Proper handling of 1-based/0-based indices
    2. Accurate conflict detection using full criteria requirements
    3. Realistic risk calculation based on achievable recommendations
    """

    rules_contains_changeable = []
    
    for i in changable_feature_ordinal:
        rules_contains_changeable.extend(rules_df[rules_df[rule_column].str.contains(i)].index.tolist())    
    rules_contains_changeable = sorted(list(set(rules_contains_changeable)))

        

    uncontrolable_met =[]
    uncontrolable_Cof = 0
    for u in uncontrolable_indices:
        if check_rule(X_test,rules_df.rule[u],patient_index):
            uncontrolable_met.append(u)
            uncontrolable_Cof+=rules_df.coef[u]

    
    def calculate_risk(data_vector, intercept):
        """Calculate sigmoid risk score"""
        return float(1 / (1 + np.exp(-(intercept + np.dot(data_vector,rules_df.coef )))))
    
    if base_0 is False:
        return print('Make sure all inputs are in base 0 index')
        
    else:
        

        # Convert all indices to 0-based for internal processing
        x_indices_0based = [x for x in x_indices]
        z_indices_0based = [z for z in z_indices]
        Q_indices_0based = [q for q in Q_indices]
        U_indices_0based = [u for u in U_indices]

        # Initial state analysis
        patient_data = embedded_X_test.loc[patient_index].values
        initial_risk = calculate_risk(patient_data, intercept )

        # Validate recommendations against constraints
        conflicts = []
        achievable_rules = []

        # Check criteria constraints first
        invalid_z = (set(Q_indices_0based).intersection(z_indices_0based) or not set(U_indices_0based).issubset(z_indices_0based))

        if invalid_z:
            print("Invalid z_indices - violates Q/U constraints")
            return None

        #  Check rule feasibility using A matrix
        for rule_idx in x_indices_0based:
            # Get criteria required by this rule (0-based)
            required_criteria = np.where(A[rule_idx] == 1)[0].tolist()

            # Check if all required criteria are in z_indices
            if not set(required_criteria).issubset(z_indices_0based):
                conflicts.append(rule_idx)  
            else:
                achievable_rules.append(rule_idx)

        # Apply achievable recommendations
        recommended_data = patient_data.copy()
        if mode == 'STRR':
            recommended_data[rules_contains_changeable] = 0
            recommended_data[achievable_rules] = 1
            
        elif mode == 'LTRR':
            recommended_data[:len(lifstyle_rule_indices_org)] = 0  # The first lifestyle rules
            recommended_data[achievable_rules] = 1
        else:
            print('NO mode is Selected')
            no_mode #this show error


        new_to_met = []
        new_to_not_met = []

        # Update rules
        for rule in range(len(patient_data)):
            if patient_data[rule] == 0 and recommended_data[rule] == 1:
                new_to_met.append(rule)
            elif patient_data[rule] == 1 and recommended_data[rule] == 0:
                new_to_not_met.append(rule)

        
        
        # Calculate realistic new risk
        new_risk = calculate_risk(recommended_data, intercept)

        # Check non-selected rules other than uncontrollable
        all_rules = set(range(len(B)))
        non_selected_rules = all_rules - set(x_indices_0based)

        fully_met_non_selected = [
            rule for rule in non_selected_rules  #
            if set(np.where(A[rule] == 1)[0]).issubset(z_indices_0based)
        ]
        if len(conflicts) != 0:
            print(f'Num of conflicts{conflicts}')

        org_met = sum(patient_data)
        risk_changed = new_risk - initial_risk
        short_report =f'Original met rules: {org_met} | New met rules: {len(achievable_rules)}\n Original prediction: {initial_risk:.4f}\n New prediction: {new_risk:.4f}\n Rules originally met but not met after changes: {len(new_to_not_met)}\n New rules met that were not originally met: {len(new_to_met)}'
        if print_results:
            # Print results
            print(f"Original met rules: {org_met} | New met rules: {len(achievable_rules+uncontrolable_met)}")
            print(f"Original prediction: {initial_risk:.4f}")
            print(f"New prediction: {new_risk:.4f}")
            print(f"Rules originally met but not met after changes: {len(new_to_not_met)}")
            print(f"New rules met that were not originally met: {len(new_to_met)}")
            
            if conflicts:
                print(f"Conflicts detected in rules {len(conflicts)}: {conflicts}")
            
            if risk_changed != 0:
                print(f"Risk {'increased' if risk_changed > 0 else 'decreased'} by: {abs(risk_changed):.4f}")
            else:
                print("No change in risk or prediction")

        
        return initial_risk,new_risk,conflicts,achievable_rules,fully_met_non_selected,new_to_met,new_to_not_met,risk_changed,org_met,short_report,uncontrolable_met,uncontrolable_Cof


def extract_criterias(rule_df, column_name='rule'): #Extract unique criterias from a DataFrame containing rules

    if column_name not in rule_df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")

    criterias = {}
    for rule in rule_df[column_name]:
        conditions = [cond.strip() for cond in rule.split('and')]
        for condition in conditions:
            criterias[condition.lower()] = condition  # Map lowercased condition to original

    # Sort based on lowercased keys but return original format values
    return [criterias[key] for key in sorted(criterias)]
    
def retrieve_feature_values(final_recommendations, features_to_retrieve):
    """
    Retrieve feature values from the final recommendation DataFrame.

    Args:
    - final_recommendation_std (DataFrame): The DataFrame with recommendations.
    - features_to_retrieve (list): A list of features to retrieve values for.

    Returns:
    - dict: A dictionary of feature values.
    """
    # Check if final_recommendations is empty

    if final_recommendations is None or final_recommendations.empty:

        print('Empty Recommendation')
        return {feature: '((No change-empty rec))' for feature in features_to_retrieve}
    feature_values = {}
    for feature in features_to_retrieve:
        if feature in final_recommendations['Feature'].values:
            # Retrieve the recommended value for the feature
            recommended_value = final_recommendations.loc[final_recommendations['Feature'] == feature, 'Recommended'].iloc[0]
            #Added Ave recommended value
            recommended_avr_value = final_recommendations.loc[final_recommendations['Feature'] == feature, 'Rounded Recommended'].iloc[0]
            feature_values[feature] = [recommended_value,recommended_avr_value]
        else:
            feature_values[feature] = 'No change'
    return feature_values

def check_infeasibility(results_text):
    """Returns True if GAMS/CPLEX reports integer infeasibility."""
    if results_text is None:
        return False
    infeasible_patterns = [
        "MODEL STATUS      10",        # Integer Infeasible
        "MODEL STATUS       5",        # Locally Infeasible
        "integer infeasible",
        "Problem is integer infeasible",
        "Integer Infeasible"
    ]
    return any(p in results_text for p in infeasible_patterns)
    
def generate_and_submit_gams_job(patient_id, mode, rule_df,U, Q, B, C, P, N, A, S, V, D, G, H, gams_file_path,Criterias ,X_test,max_retries = 5,wait_seconds=15):
    
    # Record start time
    start_time = datetime.now()
    submission_time = start_time.strftime("%I:%M:%S %p")
    # Generate GAMS file content
    ws = GamsWorkspace()
    new_parameters = (
        f"Set i /1*{len(B)}/, j /1*{len(Criterias)}/;\n"
        f"alias (j,jj);\n"
        f"parameter U(j) {list_to_gams_param(U)};\n"
        f"parameter Q(j) {list_to_gams_param(Q)};\n"
        f"parameter B(i) {list_to_gams_param(B)};\n"
        f"parameter C(i) {list_to_gams_param(C)};\n"
        f"parameter P(j) {list_to_gams_param(P)};\n"
        f"parameter N(j) {list_to_gams_param(N)};\n"

        f"Table A(i,j) \n {matrix_to_gams_table(np.array(A))};\n"
        f"Table S(i,j) \n {matrix_to_gams_table(np.array(S))};\n"

        f"Table V(j,jj) \n {matrix_to_gams_table(np.array(V))};\n"


        f"Table D(j,jj) \n {matrix_to_gams_table(np.array(D))};\n"

        f"Table G(j,jj) \n {matrix_to_gams_table(np.array(G))};\n"
        f"Table H(j,jj) \n {matrix_to_gams_table(np.array(H))};\n"
    )
    #        f"Table F(i,j) \n {matrix_to_gams_table(np.array(F))};\n"

    gams_content = """
    binary variable x(i);
    binary variable z(j);
    binary variables activate(j);
    binary variables deactivate(j);

    Free Variable o;
    
    """
    
    gams_content += """
    equation obj;
    equation constr1(i);
    equation constr2(i);
    equation constr3(i,j);
    equation constr4(j);
    equation constr5(j,jj);
    equation constr6(j,jj);
    equation constr7(j,jj);
    equation constr8(j,jj);
    equation constr9(j);  
    equation constr10(j);
    equation link_activate(j);
    equation link_deactivate(j);

    
    
    
    
    obj.. sum(i, B(i)*x(i))+0.000001*sum(j,activate(j)+deactivate(j))  =E= o;
    constr1(i).. C(i)*x(i) =L= sum(j, A(i,j)*z(j));
    constr2(i).. sum(j, A(i,j)*z(j))=L= C(i)-1+x(i);
    constr3(i,j).. z(j) =G= A(i,j)*x(i);
    constr4(j).. sum(i, A(i,j)*x(i)) =L= P(j)*z(j);
    constr5(j,jj).. z(j) - z(jj) =G= V(j,jj)-1;
    constr6(j,jj).. z(j) - z(jj) =l= 1 - D(j,jj);
    constr7(j,jj).. z(j)+z(jj) =G= G(j,jj);
    constr8(j,jj).. z(j)+z(jj) =L= 2-H(j,jj);
    constr9(j).. z(j) =L= 1 - Q(j);
    constr10(j).. z(j) =G= U(j);
    link_activate(j).. activate(j) =G= z(j) - N(j);
    link_deactivate(j).. deactivate(j) =G= N(j) - z(j);
    
    
    options LP = Cplex ;
    model m /all/;
    solve m using mip minimizing o;
    display z.l, x.l, o.l, activate.l, deactivate.l;
;
    """


    # Combine new parameters with the fixed GAMS content
    updated_content = new_parameters + "\n" + gams_content

    # Write the updated content to the GAMS file
    try:
        with open(gams_file_path, 'w') as file:
            file.write(updated_content)
    except IOError as e:
        print(f"Error writing to file {gams_file_path}: {e}")
        
        
    with open(gams_file_path, 'r') as file:
        gams_model = file.read()

        # Create an XML string for the NEOS job request
        xml_string = f"""
        <document>
            <category>lp</category>
            <solver>cplex</solver>
            <inputType>GAMS</inputType>
            <model><![CDATA[{gams_model}]]></model>
            <email>youremail@example.com</email>
        </document>
        """
        # Connect to NEOS with retry
        neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333")

        # Submit the job with retry for initial connection glitches
        submit_attempts = 0
        jobNumber, password = None, None
        while submit_attempts < max_retries:
            submit_attempts += 1
            try:
                (jobNumber, password) = neos.submitJob(xml_string)
                break
            except Exception as e:
                print(f"Submit attempt {submit_attempts} failed: {e}")
                time.sleep(wait_seconds)
                neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333")

        if jobNumber is None or password is None:
            print(f"Failed to submit job to NEOS for patient {patient_id}")
            return None, None, None, [], [], [], [], None, 'Submission Failed'
        
        # Immediately log job info with submission time
        print(f"\nJob submitted for patient {patient_id} at {submission_time}")
        print(f"Job Number: {jobNumber}")
        print(f"Password: {password}")


    """
    Retrieves and saves the NEOS output, with a retry mechanism.
    """
 
    # Initialize tracking variables
    results = None
    final_status = None
    attempts = 0
    status_history = []


    # Retry loop with timing
    start_loop_time = datetime.now()
    while attempts < max_retries:
        attempts += 1
        current_time = datetime.now().strftime("%I:%M:%S %p")
        try:
            status = neos.getJobStatus(jobNumber, password)
            status_history.append(f"Attempt {attempts} ({current_time}): {status}")
            
            while status in ['Running', 'Waiting']:
                print(f"[{current_time}] Attempt {attempts}: {status}")
                time.sleep(wait_seconds)
                status = neos.getJobStatus(jobNumber, password)
                current_time = datetime.now().strftime("%I:%M:%S %p")

            if status == 'Done':
                results = neos.getFinalResults(jobNumber, password).data.decode()
                final_status = 'Done'
                break
            else:
                print(f"[{current_time}] Unexpected status: {status}")
                final_status = status

        except Exception as e:
            current_time = datetime.now().strftime("%I:%M:%S %p")
            print(f"[{current_time}] Attempt {attempts} error: {str(e)}")
            status_history.append(f"Attempt {attempts} ({current_time}): Error - {str(e)}")
            if attempts >= max_retries:
                final_status = 'Max retries exceeded'
            # Refresh proxy to reset any broken sockets or stale DNS cache
            try:
                neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333")
            except Exception:
                pass
            time.sleep(wait_seconds)

    # Calculate total duration
    end_time = datetime.now()
    total_duration = end_time - start_time

    # Check if the job is done and print the message only once
    if final_status == 'Done':
        print(f'Optimization job for {patient_id} is Done')
    else:
        print(f"Job for {patient_id} did not complete successfully ({final_status}).")

    """
    Extract indices for 'z' and 'x' variables where the value is 1.000 in NEOS_output.
    """
    # Check for integer infeasibility first, then extract values
    is_infeasible = check_infeasibility(results)

    if is_infeasible:
        print(f"⚠️  WARNING: Patient {patient_id} — Model returned INTEGER INFEASIBLE. "
              f"No treatment selection possible for this patient's data profile.")
        z_values, x_values, o_value, z_indices, x_indices = [], [], None, [], []
        final_status = 'Infeasible'

    elif results:
        z_values, x_values, o_value = extract_neos_values(results)
        z_indices = [i - 1 for i in z_values]
        x_indices = [i - 1 for i in x_values]

    else:
        z_values, x_values, o_value, z_indices, x_indices = [], [], None, [], []

    # Print final status report
    print(f"\nPatient {patient_id} Process Summary: Start Time: {submission_time}, "
          f"End Time: {end_time.strftime('%I:%M:%S %p')}, "
          f"Total Duration: {total_duration}, "
          f"Final Status: {final_status}")
    ## End
    return jobNumber, password, results, z_values, z_indices, x_values, x_indices, o_value, final_status










def get_discretization_mapping(df, continuous_vars, n_bins=5):
    df_discretized = df.copy()
    discretization_thresholds = {}

    for var in continuous_vars:
        if var in df.columns:
            # Create bin edges based on quantiles
            bin_edges = pd.qcut(df[var], q=n_bins, duplicates='drop', retbins=True)[1]
            
            # Store the bin edges in the dictionary
            discretization_thresholds[var] = bin_edges.tolist()
            
            # Discretize the variable
            df_discretized[f"{var}_ordinal"] = pd.cut(df[var], bins=bin_edges, labels=False, include_lowest=True)
            
            # Add 1 to shift the range from 0-4 to 1-5
            df_discretized[f"{var}_ordinal"] += 1
            
            # Drop the original continuous variable
            df_discretized = df_discretized.drop(columns=[var])
        else:
            print(f"Warning: Variable '{var}' not found in the dataset.")

    return df_discretized, discretization_thresholds