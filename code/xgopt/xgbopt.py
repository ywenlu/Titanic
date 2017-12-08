# Hyperop for Xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
Xed = pickle.load(open("Xed.df", "rb"))
y = pickle.load(open("y.df", "rb"))
param_grid=    {
        'n_estimators':hp.choice('n_estimators',[5000]),
    
        'max_depth':hp.choice('max_depth',range(3,10,2)),
        'min_child_weight':hp.choice('min_child_weight',range(1,6,2)),
        
        'gamma' : hp.uniform('gamma',0,2),
    
        'subsample' : hp.uniform('subsample',0.5,1),
        'colsample_bytree' : hp.uniform('colsample_bytree',0.5,1),
        
        'reg_alpha':hp.choice('reg_alpha',[0.0001, 0.0003, 0.001,0.003,0.01,0.03,0.1]), 
        'reg_lambda':hp.choice('reg_lambda',[0.0001, 0.0003, 0.001,0.003,0.01,0.03,0.1]),
        #'verbose':hp.choice('verbose',[False]),
        #'silent':hp.choice('silent',[1])
        #'learning_rate':hp.choice('learning_rate',[0.01,0.03,0.1]),#hp.lognormal('lr',0.1,1)     
    }

xgtrain = xgb.DMatrix(Xed, label=y)

def run_trials():
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("my_model.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    def f(params):
        params = {'silent':True}
        cvresult = xgb.cv(params, xgtrain, num_boost_round=5000, nfold=10, metrics=['rmse'], early_stopping_rounds=50, seed=1301,verbose_eval=False)
        score = cvresult['test-rmse-mean'].min()
        print (score,cvresult.shape[0])
        return {'loss':score, 'status': STATUS_OK}

    best = fmin(f, param_grid, algo=tpe.suggest, max_evals=max_trials, trials=trials)

    #print("Best:", best)
    with open("my_model.hyperopt", "wb") as f:
        pickle.dump(trials, f)

# loop indefinitely and stop whenever you like
while True:
    run_trials()
