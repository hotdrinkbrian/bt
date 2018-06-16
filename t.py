#import ROOT
#import sklearn
import numpy as np
#import keras
from keras.models import Sequential
from keras.layers import Dense

path = '/home/brian/datas/v4'
bkg_name = 'QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimed.root'
sgn_name = 'VBFH_HToSSTobbbb_MH-125_MS-40_ctauS-500_TuneCUETP8M1_13TeV-powheg-pythia8_skimed.root'

bkg_name = path + bkg_name 
sgn_name = path + sgn_name 

model = Sequential([
    Dense(30, activation='relu', input_shape=(6,)),
    Dense(2, activation='softmax')
])

#model.summary()

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

data = np.random.random((1000,6))
labels = np.random.randint(2, size=(1000,2))

model.summary()

model.fit(

    data,
    labels,
    batch_size=32,
    epochs=10, verbose=2,
    shuffle=True,
    initial_epoch=0

)





