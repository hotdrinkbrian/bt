import pandas as pd
import root_numpy
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevent the plot from showing
import matplotlib.pyplot as plt
#import sys, optparse
import math
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.externals import joblib
from ROOT import TDirectory, TFile, gFile, TBranch, TTree
#from sklearn import metrics
from timeit import default_timer as timer
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Dense
#====settings==================================================================================================
test = 0
train_test_ratio = 0.5
bkg_multiple = 10
bkg_test_multiple = 100
random_seed = 19680801
roc_resolution = 0.0001
attr = ['pt','cm','phf','chf','muf','nhf','nm','elf','chm','dR_q1','dR_q2','dR_q3','dR_q4','eta']
#====settings==================================================================================================

path = '/home/brian/datas/v4/'
if test == 0:
    bkg_name = 'QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimed.root'
elif test == 1:
    bkg_name = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-0_TuneCUETP8M1_13TeV-powheg-pythia8_1j_skimed.root'
sgn_name = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_TuneCUETP8M1_13TeV-powheg-pythia8_skimed.root'

bkg_name = path + bkg_name 
sgn_name = path + sgn_name 
log = OrderedDict()

fs = TFile(sgn_name,"r")
fb = TFile(bkg_name,"r")
print 'loading tree ....'
tree_sig = fs.Get('tree44') #tree_sig = fs.Get('reconstruction;1').Get('tree')
tree_bkg = fb.Get('tree44') #tree_bkg = fb.Get('reconstruction;1').Get('tree')
print 'tree loaded ....'

lni, lno = [], []
attr_order = OrderedDict()
for s in enumerate(attr):
    lni.append( 'Jet1s.' + s[1] )  
    lno.append( 'Jet1.' + s[1] )   
    attr_order[s[1]] = s[0]
print 'attributes from input file:'    
print lni

#set up DataFrames
print 'loading data ....'
df_sig = pd.DataFrame(root_numpy.tree2array(tree_sig, branches = lni))
df_bkg = pd.DataFrame(root_numpy.tree2array(tree_bkg, branches = lni))
print 'data completely loaded'
#print 'writing data to csv file'
#df_sig.to_csv('/home/hezhiyua/scp/sig.csv')
#df_bkg[:len( df_sig )].to_csv('/home/hezhiyua/scp/bkg.csv')
#print 'data completely written'

df_sig.columns = lno
df_bkg.columns = lno

df_sig['is_signal'] = 1
df_bkg['is_signal'] = 0

#drop events with values = -1
sig_dropone = df_sig['Jet1.pt'] != -1 
bkg_dropone = df_bkg['Jet1.pt'] != -1
df_sig = df_sig[:][sig_dropone] 
df_bkg = df_bkg[:][bkg_dropone]

print 'num of entries for bkg:'
print len(df_bkg)
print 'num of entries for sig:'
print len(df_sig)

#------------------------------------------------------------
if len(df_bkg) >= bkg_test_multiple * len(df_sig):
    df_test_sig = df_sig[:int( len(df_sig) * (1-train_test_ratio) )]
    df_sig      = df_sig[len(df_test_sig):]  
    df_test_bkg = df_bkg[:int( len(df_test_sig) * bkg_test_multiple )] 
    df_bkg      = df_bkg[len(df_test_bkg):(  1 + len(df_test_bkg) + int( len(df_sig) * bkg_multiple )  )]
    print len(df_test_sig)
    print len(df_sig)
    print len(df_test_bkg)
    print len(df_bkg) 
else:
    print 'data not enough!'  
    train_test_ratio = 0.7
    df_test_sig = df_sig[:int( len(df_sig) * (1-train_test_ratio) )]
    df_sig      = df_sig[len(df_test_sig):]  
    df_test_bkg = df_bkg[:int( len(df_test_sig) * 1 )] 
    df_bkg      = df_bkg[len(df_test_bkg):]
    print len(df_test_sig)
    print len(df_sig)
    print len(df_test_bkg)
    print len(df_bkg) 
#------------------------------------------------------------

# Fixing random state for reproducibility
np.random.seed(random_seed)
df = pd.concat([df_sig, df_bkg], ignore_index=True)
df = df.iloc[np.random.permutation(len(df))]
#df = df.iloc[np.random.RandomState(seed=44).permutation(len(df))]
df_train = df
df_ts = pd.concat([df_test_sig, df_test_bkg], ignore_index=True)
df_test_orig = df_ts.iloc[np.random.permutation(len(df_ts))]
#df_test_orig = df_ts.iloc[np.random.RandomState(seed=44).permutation(len(df_ts))]


"""
bkg_multiple = 1
train_test_ratio = 0.75
df = pd.concat([df_sig, df_bkg[:len(df_sig) * bkg_multiple]], ignore_index=True)
df = df.iloc[np.random.permutation(len(df))]
df_train = df[:int( len(df) * train_test_ratio )]
df_test_orig = df[int( len(df) * train_test_ratio ):]
"""


df_train = np.asarray(df_train)
df_test = np.asarray(df_test_orig)


#------------------functions----------------------------------------------------
def dict2str(dct):
    string = ''
    for i in dct:
        temp = dct[i]
        string = string + str(i) + ': ' + '\n' + str(temp) + '\n'
    return string
#------------------functions----------------------------------------------------


# take in all the attributes info
X_train, y_train = df_train[:,(1,2,3,4,5,6,7,8)], df_train[:,len(attr)] 
X_test, y_test   = df_test[:,(1,2,3,4,5,6,7,8)] , df_test[:,len(attr)] 
#X_train, y_train = df_train[:,(3,5)], df_train[:,len(attr)]
#X_test, y_test = df_test[:,(3,5)], df_test[:,len(attr)]


from keras import utils
Y_train = utils.to_categorical(y_train)


model = Sequential([
    Dense(30, activation='relu', input_shape=(8,)),
    Dense(30, activation='relu'),
    Dense(30, activation='relu'),
    Dense(2, activation='softmax')
])


model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#data = np.random.random((100,8))
#labels = np.random.randint(2, size=(100,2))
model.summary()

history = model.fit(

    X_train, #data,
    Y_train, #labels,
    batch_size=400,
    epochs=60, verbose=2,
    shuffle=True,
    initial_epoch=0

)


y_pred = model.predict_proba(X_test)

Y_test = utils.to_categorical(y_test)


from sklearn import metrics
auc = metrics.roc_auc_score(Y_test, y_pred)

y_pred_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)

print 'AUC:'
print auc

fig = plt.figure()
plt.plot(fpr,tpr,label="bdt, auc="+str(auc))
plt.legend(loc=4)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.grid(True)
plt.title('ROC')
plt.ylabel('false pos rate')
plt.xlabel('true pos rate')
fig.savefig('./roc/roc2.pdf') #"roc.pdf", bbox_inches='tight'

print 'fit log:'
print history
