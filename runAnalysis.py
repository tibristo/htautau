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
from scipy.optimize import minimize

if len(sys.argv) < 2:
    print 'not enough args supplied!  Need to specify which method'
    sys.exit()

accuracy = 0.05
ams_best = 0.0
vars_best = ''
#file_dir = '/scratch/s1214155/htautau/'
file_dir = '/media/win/higgscomp/'
#file_dir = '/media/swap/Htautau/'
logfile = open(file_dir+sys.argv[1]+'runAnalysis.log','w')
nEvents = 0
solutionFile = file_dir+"solutions.csv"
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
    global file_dir, nEvents
    f = open(file_dir+fname+'.out','w')
    # need to figure out the rankorder still
   
    f.write('EventId,RankOrder,Class\n')
    count = 1
    for x in xrange(0,len(cls)):
        f.write(str(eventid[x])+','+str(count)+','+str(cls[x])+'\n')
        count+=1
    nEvents = count-1
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
def runDTree(depth):#, filename):
    global file_dir, nEvents, solutionFile
    #filename = 
    clf = tree.DecisionTreeClassifier(max_depth=depth[0])
    print "DecisionTreeClassifier training"
    clf = clf.fit(sigtr[train_input].values, sigtr['Label'].values)
    print "DecisionTreeClassifier testing"
    clf_pred = clf.predict(sigtest[train_input].values)
    solnFile(filename,clf_pred,sigtest['EventId'].values)#
    print "DecisionTreeClassifier finished"

# Ensemble methods http://scikit-learn.org/stable/modules/ensemble.html
# AdaBoostClassifier
def runAdaBoost(arr):#depth, n_est,  lrn_rate=1.0): # removing filename for the scipy optimise thing '''filename,'''
    #ada = AdaBoostClassifier(n_estimators=100)
    global file_dir, nEvents, solutionFile
    depth = int(arr[0]*10)
    n_est = int(arr[1]*100)
    lrn_rate = arr[2]
    fname = 'ada_dep'+str(depth)+'_est'+str(n_est)+'_lrn'+str(lrn_rate)
    filename = fname
    ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth),
                             algorithm="SAMME",
                             n_estimators=n_est)#,n_jobs=4)
    print "AdaBoost training"
    ada.fit(sigtr[train_input].values,sigtr['Label'].values)
    print "AdaBoost testing"
    ada_pred = ada.predict(sigtest[train_input].values)
    solnFile(filename,ada_pred,sigtest['EventId'].values)#
    print "AdaBoost finished"
    # added for teh scipy optimise thing
    ams_score = ams.AMS_metric(solutionFile, file_dir+fname+'.out', nEvents)
    logfile.write(fname + ': ' + str(ams_score))
    return -1.0*float(ams_score) # since we are minimising

# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#example-ensemble-plot-adaboost-multiclass-py
def runAdaReal(arr):#depth, n_est, filename, lrn_rate=1.0):
    global file_dir, nEvents, solutionFile
    depth = int(arr[0]*10)
    n_est = int(arr[1]*100)
    lrn_rate = arr[2]
    filename =  'adar_dep'+str(depth)+'_est'+str(n_est)+'_lrn'+str(lrn_rate) # low
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
    ams_score = ams.AMS_metric(solutionFile, file_dir+filename+'.out', nEvents)
    logfile.write(filename+': ' + str(ams_score))
    return -1.0*float(ams_score)
                  

# GradientBoostingClassifier
def runGDB(arr):#depth, n_est, filename, lrn_rate=1.0):
    global file_dir, nEvents, solutionFile
    depth = int(arr[0]*10)
    n_est = int(arr[1]*100)
    lrn_rate = arr[2]
    filename =  'gdb_dep'+str(depth)+'_est'+str(n_est)+'_lrn'+str(lrn_rate) # low
    print "GDB training"
    gdb = GradientBoostingClassifier(init=BaseTree(tree.DecisionTreeClassifier(max_depth=depth)),n_estimators=n_est, learning_rate=lrn_rate, random_state=0).fit(sigtr[train_input].values, sigtr['Label'].values)
    print "GDB testing"
    gdb_pred = gdb.predict(sigtest[train_input].values)
    solnFile(filename,gdb_pred,sigtest['EventId'].values)#
    print "GDB finished"
    ams_score = ams.AMS_metric(solutionFile, file_dir+filename+'.out', nEvents)
    logfile.write(filename+': ' + str(ams_score))
    return -1.0*float(ams_score)

# ANN - http://scikit-learn.org/stable/auto_examples/plot_rbm_logistic_classification.html#example-plot-rbm-logistic-classification-py
def runRBM(arr, clsfr):#iters, lrn_rate, logistic_c_val, logistic_c_val2, n_comp, filename):
    global file_dir, nEvents, solutionFile
    iters = int(arr[0]*10)
    lrn_rate = arr[1]
    logistic_c_val = arr[2]*1000.0
    logistic_c_val2 = arr[3]*100.0
    n_comp = int(arr[4]*100)
    filename = 'rbm_iter'+str(iters)+'_logc'+str(log_c_val)+'_logcc'+str(log_c_val2)+'_lrn'+str(learn_rate)+'_nc'+str(n_comp)# low
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
    if clsfr == 0:
        clsnn_pred=classifier.predict(sigtest[train_input].values)
        solnFile('clsnn_'+filename,clsnn_pred,sigtest['EventId'].values)#,bkgtest)
        ams_score = ams.AMS_metric(solutionFile, file_dir+filename+'.out', nEvents)
        logfile.write(filename+': ' + str(ams_score))
    
    elif clsfr == 1:
        log_cls_pred = logistic_classifier.predict(sigtest[train_input].values)
        solnFile('lognn_'+filename,log_cls_pred,sigtest['EventId'].values)#,bkgtest)
        ams_score = ams.AMS_metric(solutionFile, file_dir+'lognn_'+filename+'.out', nEvents)
        logfile.write('lognn ' + filename+': ' + str(ams_score))
    else:
        logistic_classifier_tx = linear_model.LogisticRegression(C=logistic_c_val2)
        logistic_classifier_tx.fit_transform(sigtr[train_input].values, sigtr['Label'].values)
        log_cls_tx_pred = logistic_classifier_tx.predict(sigtest[train_input].values)
        solnFile('lognntx_'+filename,log_cls_tx_pred,sigtest['EventId'].values)#,bkgtest)
        ams_score = ams.AMS_metric(solutionFile, file_dir+filename+'.out', nEvents)
        logfile.write('lognntx '+ filename+': ' + str(ams_score))

    return -1.0*float(ams_score)


    

iters = 0.20 # * 10 in method
log_cval = 6.0 # * 1000 in method
log_cval2 = 1.0# * 100 in method
n_comps = 0.200 # *100 in method
# run maximise scores some number of times - 10 maybe?
method = sys.argv[1]

lrn_rate = 1.0
n_est = 2.00 #will *100 in method
depth = 0.6 # will *10 in method

if sys.argv[1] == 'AdaBoost':
    x0ada = np.array([depth,n_est,lrn_rate])
    res_ada = minimize(runAdaBoost, x0ada, method='nelder-mead',options={'xtol': 1e-2, 'maxfev':100, 'disp': True})
    print res_ada.x
    logfile.write(res_ada.x)

elif sys.argv[1] == 'AdaReal':
    x0adar = np.array([depth,n_est,lrn_rate])
    res_r = minimize(runAdaReal, x0adar, method='nelder-mead',options={'xtol': 1e-2, 'maxfev':100,'disp': True})
    print res_r.x
    logfile.write(res_r.x)

elif sys.argv[1] == 'GDB':
    x0gdb = np.array([depth,n_est,lrn_rate])
    res_gdb = minimize(runGDB, x0gdb, method='nelder-mead',options={'xtol': 1e-2, 'maxfev':100, 'disp': True})
    print res_gdb.x
    logfile.write(res_gdb.x)

#iters, lrn_rate, logistic_c_val, logistic_c_val2, n_comp, filename):
elif sys.argv[1] == 'RBM0':
    x0rbm = np.array([iters,lrn_rate,log_cval, log_cval2, n_comps])
    res_rbm = minimize(runRBM, x0rbm, args=(0), method='nelder-mead',options={'xtol': 1e-2,  'maxfev':100,'disp': True})
    print res_rbm.x
    logfile.write(res_rbm.x)

#iters, lrn_rate, logistic_c_val, logistic_c_val2, n_comp, filename):
elif sys.argv[1] == 'RBM1':
    x0rbm1 = np.array([iters,lrn_rate,log_cval, log_cval2, n_comps])
    res_rbm1 = minimize(runRBM, x0rbm1, args=(1), method='nelder-mead',options={'xtol': 1e-2, 'maxfev':100, 'disp': True})
    print res_rbm1.x
    logfile.write(res_rbm1.x)

#iters, lrn_rate, logistic_c_val, logistic_c_val2, n_comp, filename):
elif sys.argv[1] == 'RBM2':
    x0rbm2 = np.array([iters,lrn_rate,log_cval, log_cval2, n_comps])
    res_rmb2 = minimize(runRBM, x0rbm2, args=(2), method='nelder-mead',options={'xtol': 1e-2,'maxfev':100, 'disp': True})
    print res_rbm2.x
    logfile.write(res_rbm2.x)


logfile.close()
