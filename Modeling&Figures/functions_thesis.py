import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import numbers
from dateutil import parser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.inspection import permutation_importance
import collections
import time
import ast
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NeighbourhoodCleaningRule, EditedNearestNeighbours, RandomUnderSampler
from collections import Counter
import re
import nltk
from xgboost import XGBClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from emoji import UNICODE_EMOJI
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import random
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def retweet_group(data, groups):
    for item in groups:
        left = groups[item][0]
        right = groups[item][1]
        
        # check if datapoint belongs to group
        if (data >= left) & (data < right):
            return item
        
def check_emoticon(text):
    counter = 0    
    for item in text.split():
        if item in UNICODE_EMOJI['en']:
            counter = counter + 1
    return counter

def preprocessing(data):
    # add length in characters of tweet as feature
    data['tweet_char_len'] = data.apply(lambda row: len([item for item in row.text]), axis = 1)

    # assign tweets to retweet class
    groups = {1 : [0, 1], 2 : [1, 10], 3  : [10, 100], 4 : [100, 1000], 5 : [1000, 1000000000]}
    data['retweet_class'] = data.apply(lambda row: retweet_group(row['public_metrics.retweet_count'], groups), axis = 1)
    print("count retweet classes AFTER", data.groupby('retweet_class').count()['id'])

    # make feature ratio followers and following 
    data['ratio_follow'] = data.apply(lambda row: row['public_metrics.followers_count'] / row['public_metrics.following_count'] if (row['public_metrics.following_count'] > 0) else row['public_metrics.followers_count'], axis = 1)

    # create hour of the day feature
    data['hour_day'] = data.apply(lambda row: parser.parse(row.created_at).hour, axis = 1)
    
    male_items = ['male', 'mostly_male']
    female_items = ['female', 'mostly_female']
    data['sex_generalized'] = data.apply(lambda row: 1 if (row.sex in male_items) else (-1 if (row.sex in female_items) else 0), axis = 1)
 
    data['hashtag'] = data['entities.hashtags'].isnull()
    data['urls'] = data['entities.urls'].isnull()
    
    data['log_followers'] = data['public_metrics.followers_count'].apply(np.log1p)
    data['log_following'] = data['public_metrics.following_count'].apply(np.log1p)
    data['log_listed'] = data['public_metrics.listed_count'].apply(np.log1p)
    data['log_tweetcount'] = data['public_metrics.tweet_count'].apply(np.log1p)
    
    data['hashtag_count'] = data.apply(lambda row: len(ast.literal_eval(row['entities.hashtags'])) if pd.isna(row['entities.hashtags']) == False else 0, axis = 1) 
    data['mention_count'] = data.apply(lambda row: len(ast.literal_eval(row['entities.mentions'])) if pd.isna(row['entities.mentions']) == False else 0, axis = 1) 
    data['urls_count'] = data.apply(lambda row: len(ast.literal_eval(row['entities.urls'])) if pd.isna(row['entities.urls']) == False else 0, axis = 1)

    data['viral'] = data.apply(lambda row: 1 if row['public_metrics.retweet_count'] > 100 else 0, axis = 1)

    data['organization'] = data.apply(lambda row: 1 if (row.sex_generalized == 0) & (row.verified == True) else 0, axis = 1)
    
    sid = SentimentIntensityAnalyzer()
    data['sentiment'] = data.apply(lambda row: sid.polarity_scores(row.text)['compound'], axis = 1)
    
    data['emoji_count'] = data.apply(lambda row: check_emoticon(row.text), axis = 1)
    data['verified'] = data.apply(lambda row: 1 if row.verified == True else 0, axis = 1)
    
    return data

def get_scores(y_true, y_pred):
    metrics = dict()
    metrics['report'] = classification_report(y_true, y_pred, labels = [0, 1], digits = 3)
    metrics['roc_auc_score'] = roc_auc_score(y_true, y_pred, labels = [0, 1], average = 'macro')
    metrics['matrix'] = confusion_matrix(y_true, y_pred, labels = [0, 1])
    return metrics

def cross_validation_train(model, X, Y, resample, scale):
    features = X.keys().drop('public_metrics.retweet_count')
    
    number = 1
    metrics = dict()
    importances = collections.defaultdict(list)
    
    # make stratefied Kfold for cv (takes class balance into account)
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    skf.get_n_splits(X, Y)
    
    start_time = time.time()
    
    # perform cross val
    for train_index, test_index in skf.split(X, Y):
        print("train model number: ", number)
        
        # split data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        if resample != False:
            if isinstance(resample, int):
                print("removing train instances from:", resample)
                X_train = X_train.reset_index().drop(columns = ['index'])
                Y_train = Y_train.reset_index().drop(columns = ['index'])['viral']
                ind = X_train.index[(X_train['public_metrics.retweet_count'] < resample) | (X_train['public_metrics.retweet_count'] > 100)].tolist()
                X_train = X_train.iloc[ind, :]
                Y_train = Y_train.iloc[ind] 
            else:
                print("old: ", Counter(Y_train))
                X_train, Y_train = resample.fit_resample(X_train, Y_train)
        
        X_train = X_train.drop(columns = ['public_metrics.retweet_count'])
        X_test = X_test.drop(columns = ['public_metrics.retweet_count'])
        
        if scale == True:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        print("train: ", Counter(Y_train))
        print("test: ", Counter(Y_test))
        
        # fit model
        MODEL = model.fit(X_train, Y_train)
        
        # get evaluation and importances
        y_pred = MODEL.predict(X_test)
        scores = get_scores(Y_test, y_pred)
        metrics[number] = scores
        print(scores['report'])
        
        imp = permutation_importance(model, X_test, Y_test, n_repeats = 10, random_state = 42, scoring = 'f1_macro', n_jobs = 1)
        for item in imp:
            if item == 'importances':
                for feature, importance in zip(features, imp[item]):
                    importances[feature].append(np.mean(importance))       
        print(ConfusionMatrixDisplay.from_predictions(Y_test, y_pred))
       
        # increase number
        number = number + 1
        
    print("minutes running code: ", ((time.time() - start_time) / 60))
    
    return metrics, importances

def baseline_model(X, Y):
    number = 1
    metrics = dict()
    # make stratefied Kfold for cv (takes class balance into account)
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    skf.get_n_splits(X, Y)
     # perform cross val
    for train_index, test_index in skf.split(X, Y):
        print("train model number: ", number)
        
        # split data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        y_pred = np.zeros(len(Y_test))
        scores = get_scores(Y_test, y_pred)
        metrics[number] = scores
        print(scores['report'])
        print(ConfusionMatrixDisplay.from_predictions(Y_test, y_pred))
        number = number + 1
        
    return metrics 

def get_f1_macro(metrics_dict):
    macro_f1 = list()
    for item in metrics_dict:
        macro_f1.append(float(metrics_dict[item]['report'].split('\n')[6].split()[4]))

    print(macro_f1)
    print(round(np.mean(macro_f1),3), round(np.std(macro_f1), 3))
    
    return round(np.mean(macro_f1),3), round(np.std(macro_f1), 3)

def best_resampling(model, X, Y, resampling_methods, scaler):
    # prep everything
    cv = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    sampling_strategy = np.arange(0.02, 0.21, 0.01)
    grid = dict(resample__sampling_strategy = sampling_strategy)
    
    # store scores
    best_scores_mean = dict()
    best_scores_std = dict()
    best_ratio = dict()
    
    for resampler in resampling_methods.keys():
        resampling = resampling_methods[resampler]
        print("sampling method: ", resampler)
        
        if resampler == 'bound':
            mean = 0
            std = 0
            param = 0
            
            for item in resampling_methods[resampler]:
                scores = best_boundary(model, X, Y, item, scaler, cv)
                new_mean = np.mean(scores)
                new_std = np.std(scores)
                
                if new_mean > mean:
                    mean = round(new_mean, 3)
                    std = round(new_std, 3)
                    param = item
            best_scores_mean[resampler] = mean
            best_scores_std[resampler] = std
            best_ratio[resampler] = param

            print("mean score: %f +- %f" % (mean, std))
            print("best parameters: ", param)
            
        else:
            X_ = X.drop(columns = ['public_metrics.retweet_count'])

            if scaler == True:
                steps = [('scaler', StandardScaler()),
                         ('resample', resampling),
                         ('model', model)]
            else: 
                steps = [('resample', resampling),
                         ('model', model)]
                
            from imblearn.pipeline import Pipeline
            pipeline = Pipeline(steps)
            grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, n_jobs=3, cv=cv, scoring='f1_macro', refit = False)
            grid_result = grid_search.fit(X_, Y)
            mean = pd.DataFrame(grid_result.cv_results_).iloc[grid_result.best_index_]['mean_test_score']
            std = pd.DataFrame(grid_result.cv_results_).iloc[grid_result.best_index_]['std_test_score']
            param = grid_result.best_params_
            
            best_scores_mean[resampler] = mean
            best_scores_std[resampler] = std
            best_ratio[resampler] = param

            print("mean score: %f +- %f" % (mean, std))
            print("best parameters: ", param)

    return best_scores_mean, best_scores_std, best_ratio
 
    
def best_boundary(model, X, Y, resample, scale, cv):    
    scores = list()
    
    cv.get_n_splits(X, Y)
        
    # perform cross val
    for train_index, test_index in cv.split(X, Y):        
        # split data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        X_train = X_train.reset_index().drop(columns = ['index'])
        Y_train = Y_train.reset_index().drop(columns = ['index'])['viral']
        ind = X_train.index[(X_train['public_metrics.retweet_count'] <= resample) | (X_train['public_metrics.retweet_count'] > 100)].tolist()
        X_train = X_train.iloc[ind, :]
        Y_train = Y_train.iloc[ind]
        
        X_train = X_train.drop(columns = ['public_metrics.retweet_count'])
        X_test = X_test.drop(columns = ['public_metrics.retweet_count'])
        
        if scale == True:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        # fit model
        MODEL = model.fit(X_train, Y_train)
        
        # get evaluation and importances
        y_pred = MODEL.predict(X_test)
        scores.append(f1_score(Y_test, y_pred, average = 'macro'))

    return scores