import numpy
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder

def PreprocessingData(data_input):
    TwoDim_dataset = data_input.reshape(len(data_input),-1)
    binarizer = Binarizer(threshold=0.0).fit(TwoDim_dataset)
    TwoDim_dataset = binarizer.transform(TwoDim_dataset)
    return TwoDim_dataset

def ChuanHoaLabel(labels):
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    return y


