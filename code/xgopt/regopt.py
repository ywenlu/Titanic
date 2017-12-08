# Hyperop for sklearn regressors
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge,Lasso,ElasticNet 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

import sys


Xed = pickle.load(open("Xed.df", "rb"))
y = pickle.load(open("y.df", "rb"))

modelname = sys.argv[1]
param_grids = {
    'Ridge':{
        'alpha': hp.uniform('alpha',0,20)
    },
    'Lasso':{
        'alpha': hp.uniform('alpha',0,20)
    },
    'ENet':{
        'alpha':hp.uniform('alpha',0,1),
        'l1_ratio':hp.uniform('l1_ratio',0,1)
    },
    'RFR':{
        'n_estimators': hp.choice('n_estimators',[500]),
        'max_depth':hp.choice('max_depth',[2,4,8,12,None]),
        'max_features':hp.choice('max_features',['sqrt',None])
    },
    'GBR':{
        'n_estimators':hp.choice('n_estimators',[500]),
        'learning_rate':hp.choice('learning_rate',[0.05]),
        'max_depth':hp.choice('max_depth',[2,4,8,12,None]),
        'max_features':hp.choice('max_features',['sqrt',None]),
        'loss':hp.choice('loss',['ls','huber'])
    }
    }

models = {
    'Ridge': Ridge(),
    'Lasso':Lasso(),
    'ENet': ElasticNet(),
    'RFR':RandomForestRegressor(),
    'GBR':GradientBoostingRegressor()
    }


def run_trials():
    trials_step = 5  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        paramfile= 'para_'+ modelname + '_.hyperopt'
        trials = pickle.load(open(paramfile, "rb"))
        print("Found saved Trials! Loading Parameter "+modelname+"...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    def f(params):
        # params = {'silent':True} # Add predefined params here
        model = models[modelname]
        model.set_params(**params)
        scores = cross_val_score(model,Xed, y, scoring='neg_mean_squared_error',cv=10 )
        rmse_scores = np.sqrt(-scores)
        return {'loss':rmse_scores.mean(), 'status': STATUS_OK}

    best = fmin(f, param_grids[modelname], algo=tpe.suggest, max_evals=max_trials, trials=trials)

    #print("Best:", best)
    with open(paramfile, "wb") as f:
        pickle.dump(trials, f)
    return best, max_trials
# loop indefinitely and stop whenever you like
bestever = 9999999999999
ct = 0
while True:
    best, max_trials = run_trials()
    if best < bestever:
        bestever = best
        ct = 0
    else:
        ct = ct+1
    if ct>20:
        break
