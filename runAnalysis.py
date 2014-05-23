import HiggsBosonCompetition_AMSMetric_rev1 as ams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import tree, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

accuracy = 0.05
ams_best = 0.0
vars_best = ''
file_dir = '/scratch/s1214155/htautau/'
logfile = open(file_dir+sys.argv[1]+'runAnalysis.log','w')
# Need an adapter to give the gradientboostclassifier a decisiontree as a parameter
# https://stackoverflow.com/questions/17454139/gradientboostingclassifier-with-a-baseestimator-in-scikit-learn/19679862#19679862
class BaseTree:
    def __init__(self, est):
        self.est = est
    def predict(self, X):
        return self.est.predict_proba(X)[:,1][:,np.newaxis]
    def fit(self, X, y):
        self.est.fit(X, y)

# write out the solution file in the correct format
def solnFile(fname, cls, eventid, bkg=None):
    global file_dir
    f = open(file_dir+fname+'.out','w')
    # need to figure out the rankorder still
   
    f.write('EventId,RankOrder,Class\n')
    count = 1
    for x in xrange(0,len(cls)):
        f.write(str(eventid[x])+','+str(count)+','+str(cls[x])+'\n')
        count+=1
    if bkg:
        for x,r in bkg.iterrows():
            f.write(str(r['EventId'])+','+str(count)+',b\n')
            count+=1
    f.close()

# list of all the variables
cols = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep',
       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
       'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
       'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau',
       'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt',
       'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta',
       'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet',
       'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta',
       'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi',
       'PRI_jet_all_pt', 'Weight', 'Label']
# These are the variables we want to train on
train_input = ['DER_mass_MMC', 'DER_mass_transverse_met_lep',
       'DER_mass_vis', 'DER_pt_h', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
       'DER_sum_pt', 'DER_pt_ratio_lep_tau',
       'DER_met_phi_centrality', 'DER_lep_eta_centrality', 
       'PRI_jet_num']


#tr = pd.read_csv('training.csv')
tr = pd.read_csv(file_dir+'training.csv')
#test = pd.read_csv('testing.csv')
test = pd.read_csv(file_dir+'testing.csv')

# How much does it change the results if we apply presel cuts on training and test, vs test ONLY?

# Apply some basic cuts to remove backgrounds
# all failed events are already background - need to write this to file or save it somewhere
# tr=tr[(tr.varname > something) & (tr.varname < something)]
# mtw cut of 70 gev
sigtr=tr#[(tr.DER_mass_transverse_met_lep < 70) & (tr.DER_mass_vis > 30) & (tr.DER_mass_vis < 140) & ((tr.DER_mass_transverse_met_lep > 40) | (tr.DER_mass_MMC > 110))]
#bkgtr=tr[(tr.DER_mass_transverse_met_lep > 70) | (tr.DER_mass_vis < 30) | (tr.DER_mass_vis > 140) | ((tr.DER_mass_transverse_met_lep < 40) & (tr.DER_mass_MMC < 110))]


sigtest=test#[(test.DER_mass_transverse_met_lep < 70) & (test.DER_mass_vis > 30) & (test.DER_mass_vis < 140) & ((test.DER_mass_transverse_met_lep > 40) | (test.DER_mass_MMC > 110))]
#bkgtest=test[(test.DER_mass_transverse_met_lep > 70) | (test.DER_mass_vis < 30) | (test.DER_mass_vis > 140) | ((test.DER_mass_transverse_met_lep < 40) & (test.DER_mass_MMC < 110))]

# try a cut on the vis mass?
#sig = pd.concat([sig,tr[(tr.DER_mass_vis > 30) & (tr.DER_mass_vis < 140)]])
# remove z->tautau enriched control region
#sig = pd.concat([sig,tr[(tr.DER_mass_transverse_met_lep > 40) & (tr.DER_mass_mmc > 110)]])



# DecisionTreeClassifier http://scikit-learn.org/stable/modules/tree.html#tree-classification
def runDTree(depth, filename):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    print "DecisionTreeClassifier training"
    clf = clf.fit(sigtr[train_input].values, sigtr['Label'].values)
    print "DecisionTreeClassifier testing"
    clf_pred = clf.predict(sigtest[train_input].values)
    solnFile(filename,clf_pred,sigtest['EventId'].values)#
    print "DecisionTreeClassifier finished"

# Ensemble methods http://scikit-learn.org/stable/modules/ensemble.html
# AdaBoostClassifier
def runAdaBoost(depth, n_est, filename, lrn_rate=1.0):
    #ada = AdaBoostClassifier(n_estimators=100)
    ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth),
                             algorithm="SAMME",
                             n_estimators=n_est)#,n_jobs=4)
    print "AdaBoost training"
    ada.fit(sigtr[train_input].values,sigtr['Label'].values)
    print "AdaBoost testing"
    ada_pred = ada.predict(sigtest[train_input].values)
    solnFile(filename,ada_pred,sigtest['EventId'].values)#
    print "AdaBoost finished"

# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#example-ensemble-plot-adaboost-multiclass-py
def runAdaReal(depth, n_est, filename, lrn_rate=1.0):
    bdt_real = AdaBoostClassifier(
        tree.DecisionTreeClassifier(max_depth=depth),
        n_estimators=n_est,
        learning_rate=lrn_rate)
    print "AdaBoostReal training"
    bdt_real.fit(sigtr[train_input].values,sigtr['Label'].values)
    print "AdaBoostReal testing"
    bdt_real_pred = bdt_real.predict(sigtest[train_input].values)
    solnFile(filename,bdt_real_pred,sigtest['EventId'].values)#
    print "AdaBoostReal finished"

# GradientBoostingClassifier
def runGDB(depth, n_est, filename, lrn_rate=1.0):
    print "GDB training"
    gdb = GradientBoostingClassifier(init=BaseTree(tree.DecisionTreeClassifier(max_depth=depth)),n_estimators=n_est, learning_rate=lrn_rate, random_state=0).fit(sigtr[train_input].values, sigtr['Label'].values)
    print "GDB testing"
    gdb_pred = gdb.predict(sigtest[train_input].values)
    solnFile(filename,gdb_pred,sigtest['EventId'].values)#
    print "GDB finished"

# ANN - http://scikit-learn.org/stable/auto_examples/plot_rbm_logistic_classification.html#example-plot-rbm-logistic-classification-py
def runRBM(iters, lrn_rate, logistic_c_val, logistic_c_val2, n_comp, filename):
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = lrn_rate #0.10#0.06
    rbm.n_iter = iters #20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = n_comp # 250
    logistic.C = logistic_c_val #6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(sigtr[train_input].values, sigtr['Label'].values)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=logistic_c_val2)#100.0
    logistic_classifier.fit(sigtr[train_input].values, sigtr['Label'].values)

    ###############################################################################
    # Evaluation
    clsnn_pred=classifier.predict(sigtest[train_input].values)
    solnFile('clsnn_'+filename,clsnn_pred,sigtest['EventId'].values)#,bkgtest)
    
    log_cls_pred = logistic_classifier.predict(sigtest[train_input].values)
    solnFile('lognn_'+filename,log_cls_pred,sigtest['EventId'].values)#,bkgtest)
    
    logistic_classifier_tx = linear_model.LogisticRegression(C=logistic_c_val2)
    logistic_classifier_tx.fit_transform(sigtr[train_input].values, sigtr['Label'].values)
    log_cls_tx_pred = logistic_classifier_tx.predict(sigtest[train_input].values)
    solnFile('lognntx_'+filename,log_cls_tx_pred,sigtest['EventId'].values)#,bkgtest)


def maximiseScores(method, learn_rate, n_est, max_depth, iters, log_cval, log_cval2, n_comps):
    global file_dir
    global logfile
    solutionFile = file_dir+"solutions.csv"
    # parameters to vary for ensemble/ tree based methods- n_estimators, max_depth, learning_rate
    # vary input variables
    # vary testing/ training with/without background cuts.  We should ideally train on all, but when running, remove with some cuts?
    # For ANN, vary these -> iters, lrn_rate, logistic_c_val, logistic_c_val2, n_comp
    params = {}
    params['learn_rate_low'] = 0.1
    params['learn_rate_high'] = 1.5
    params['learn_rate'] = learn_rate
    params['n_est_low'] = 20
    params['n_est_high'] = 500
    params['n_est'] = n_est
    params['max_depth_low'] = 1
    params['max_depth_high'] = 20
    params['max_depth'] = max_depth
    params['iters_low'] = 5
    params['iters_high'] = 50
    params['iters'] = iters
    params['log_cval_low'] = 500.0
    params['log_cval_high'] = 10000.0
    params['log_cval'] = log_cval
    params['log_cval2_low'] = 20.0
    params['log_cval2_high'] = 200.0
    params['log_cval2'] = log_cval2
    params['n_comps_low'] = 50
    params['n_comps_high'] = 500
    params['n_comps'] = n_comps
    
    global ams_best
    global accuracy
    global vars_best
    ams_prev = 0.0
    nEvents = 125000
    # want to keep an overall best score, but best for each variable too!
    max_iters = 15
    # must check that parameter is being used for that method!
    fname_low = 'notset'
    fname_high = 'notset'
    for key in params.keys():
        if key.find('low') != -1 or key.find('high') != -1:
            continue
        keylo = key+'_low'
        keyhi = key+'_high'
        itercount = 0
        optimal = False
        changedHi = False
        while not optimal and itercount < max_iters: # how many iterations to run before stopping???
            
            # need to put these in try / catch incase it crashes
            if  (not method == 'RBM') and key in ['iters','log_cval','log_cval2','n_comps']:
                optimal = True
                continue
            if method == 'RBM' and key in ['max_depth', 'n_est']:
                optimal = True
                continue

            if method == 'AdaBoost':
            #depth, n_est, filename, lrn_rate=1.0
                if not changedHi or iterations == 0:
                    params[key] = params[keylo]
                    fname_low = 'ada_dep'+str(params['max_depth'])+'_est'+str(params['n_est'])+'_lrn'+str(params['learn_rate'])
                    runAdaBoost(params['max_depth'], params['n_est'], fname_low , params['learn_rate']) # low
                if changedHi or iterations == 0:
                    params[key] = params[keyhi]
                    fname_high = 'ada_dep'+str(params['max_depth'])+'_est'+str(params['n_est'])+'_lrn'+str(params['learn_rate'])
                    runAdaBoost(params['max_depth'], params['n_est'], fname_high, params['learn_rate'])            
            elif method == 'AdaReal':
                if not changedHi or iterations == 0:
                    params[key] = params[keylo]
                    fname_low =  'adar_dep'+str(params['max_depth'])+'_est'+str(params['n_est'])+'_lrn'+str(params['learn_rate']) # low
                    runAdaReal(params['max_depth'], params['n_est'], fname_low, params['learn_rate'])
                if changedHi or iterations == 0:
                    params[key] = params[keyhi]
                    fname_high = 'adar_dep'+str(params['max_depth'])+'_est'+str(params['n_est'])+'_lrn'+str(params['learn_rate']) # high
                    runAdaReal(params['max_depth'], params['n_est'], fname_high, params['learn_rate'])
            elif method == 'GDB':
                if not changedHi or iterations == 0:
                    params[key] = params[keylo]
                    fname_low = 'gdb_dep'+str(params['max_depth'])+'_est'+str(params['n_est'])+'_lrn'+str(params['learn_rate']) # low
                    runGDB(params['max_depth'], params['n_est'], fname_low, params['learn_rate']) # low
                if changedHi or iterations == 0:
                    params[key] = params[keyhi]
                    fname_high = 'gdb_dep'+str(params['max_depth'])+'_est'+str(params['n_est'])+'_lrn'+str(params['learn_rate']) # high
                    runGDB(params['max_depth'], params['n_est'], fname_high, params['learn_rate'])
            elif method == 'RBM':
            #iters, lrn_rate, logistic_c_val, logistic_c_val2, n_comp
                if not changedHi or iterations == 0:
                    params[key] = params[keylo]
                    fname_low = 'rbm_iter'+str(params['iters'])+'_logc'+str(params['log_c_val'])+'_logcc'+str(params['log_c_val2'])+'_lrn'+str(params['learn_rate'])+'_nc'+str(params['n_comp'])# low
                    runRBM(params['iters'], params['lrn_rate'], params['log_c_val'], params['log_c_val2'], params['n_comp'], fname_low) # low
                if changedHi or iterations == 0:
                    params[key] = params[keyhi]
                    fname_high = 'rbm_iter'+str(params['iters'])+'_logc'+str(params['log_c_val'])+'_logcc'+str(params['log_c_val2'])+'_lrn'+str(params['learn_rate'])+'_nc'+str(params['n_comp']) # low
                    runRBM(params['iters'], params['lrn_rate'], params['log_c_val'], params['log_c_val2'], params['n_comp'], fname_high) # high

            try:
                print fname_high
                ams_up = ams.AMS_metric(solutionFile, file_dir+fname_high+'.out', nEvents)
            except:
                print 'could not get AMS_metric for ams_up :('
                iterations+=1
                continue
            try:
                print fname_low
                ams_do = ams.AMS_metric(solutionFile, file_dir+fname_low+'.out', nEvents)
            except:
                print 'could not get AMS_metric for ams_do :('
                iterations+=1
                continue
            if ams_up >= ams_best:
                ams_best = ams_up
                vars_best = fname_high
            if ams_do >= ams_best:
                ams_best = ams_do
                vars_best = fname_low


            if ams_up < ams_do:
                params[keyhi] = (params[keyhi]+params[keylo])/2
                changedHi = True
            else:
                params[keylo] = (params[keyhi]+params[keylo])/2
                changedHi = False
            if (1.0 - abs((float(params[keyhi] - params[keylo]))/float((params[keyhi]+params[keylo])))) % 1 <= accuracy: # within 5% of each other
                optimal = True
            # perhaps stop if the last x iterations have failed to produce better results??
            ams_prev = max(ams_up,ams_do)
            print 'ams_best: ' + str(ams_best)
            print 'ams_up: ' + str(ams_up)
            print 'ams_do: ' + str(ams_do)
            itercount +=1
            print 'Itercount: ' + str(itercount) + ' on variable ' + key
            logfile.write('Itercount: ' + str(itercount)+ ' variable: ' + key+'\n')
            logfile.write('ams_best: ' + str(ams_best)+' vars: ' + vars_best +'\n')
            logfile.write('ams_up: ' + str(ams_up)+ ' vars: ' + fname_high + '\n')
            logfile.write('ams_do: ' + str(ams_do)+ ' vars: ' + fname_low + '\n')
            logfile.write('#####\n')

    return [params['learn_rate'], params['n_est'], params['max_depth'], params['iters'], params['log_cval'], params['log_cval2'], params['n_comps']]



learn_rate = 1.0
n_est = 200
max_depth = 6
iters = 20
log_cval = 6000.0
log_cval2 = 100.0
n_comps = 200
# run maximise scores some number of times - 10 maybe?
cont = True
scores_acc = 0.02
method = sys.argv[1]
iterations = 0

while cont and iterations < 10:
    learn_rate_f, n_est_f, max_depth_f, iters_f, log_cval_f, log_cval2_f, n_comps_f = maximiseScores(method,learn_rate, n_est, max_depth, iters, log_cval, log_cval2, n_comps)
    # if we see less than... some percentage change for all values, then stop
    lrn_d = learn_rate_f/learn_rate
    cont = False
    if abs(1-lrn_d) % 1 > scores_acc: # 2% change
        learn_rate = learn_rate_f
        cont = True
    n_est_d = n_est_f/n_est
    if abs(1-n_est_d) % 1 > scores_acc:
        n_est = n_est_f
        cont = True
    max_depth_d = max_depth_f/max_depth
    if abs(1-max_depth_d)  % 1 > scores_acc:
        max_depth = max_depth_f
        cont = True
    iters_d = iters_f/iters
    if abs(1-iters_d) % 1 > scores_acc:
        iters = iters_f
        cont = True
    log_cval_d = log_cval_f/log_cval
    if abs(1-log_cval_d) % 1 > scores_acc:
        log_cval = log_cval_f
        cont = True
    log_cval2_d = log_cval2_f/log_cval2
    if abs(1-log_cval2_d) % 1 > scores_acc:
        log_cval2 = log_cval2_f
        cont = True
    n_comps_d = n_comps_f/n_comps
    if abs(1-n_comps_d) % 1 > scores_acc:
        n_comps = n_comps_f
        cont = True
    iterations += 1
    logfile.write('running iteration number '+ str(iterations)+'\n')

logfile.close()
