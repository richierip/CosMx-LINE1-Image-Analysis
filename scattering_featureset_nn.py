import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import IPython


# path_to_data = r"../wavelet_scattering_J2.csv"
# path_to_data = r"../3.15.23_synthetic_b=10_o=5_lir=15_NOBLANKS.csv"
path_to_data = r"../3.15.23_padded_J=2.csv"
seed = 42
TEST_FRACTION = 0.1


def get_training_data_plus(num_classes, cancer = "Cancer"):
    def convert_col_to_np(str_array):
        return list(np.fromstring(str_array.strip('[]'), sep= ' ', dtype = np.float32))[1:] # remove first (just lowpass)
    df = pd.read_csv(path_to_data).dropna() # Seems that there are some holes in the data, not sure why... need to investigate
    if cancer:
        wvlt = df.loc[df["Cancer?"]==cancer, "Wavelet Scattering Vector One"]
        # wvlt2 = df.loc[df["Cancer?"]==cancer, "Wavelet Scattering Vector Two"]
        # wvlt3 = df.loc[df["Cancer?"]==cancer, "Wavelet Scattering Vector Three"]
        others = df.loc[df["Cancer?"]==cancer, ["Entire cell area","DAPI Intensity Mean", "DAPI Area (px)", "Cell Width","Cell Height"]].to_numpy()
        data_labels = pd.qcut(df.loc[df["Cancer?"]==cancer,"Line1_Combined"],num_classes, labels=False).to_numpy()
    else:
        wvlt = df.loc[:, "Wavelet Scattering Vector One"]
        others = df.loc[:, ["Entire cell area","DAPI Intensity Mean", "DAPI Area (px)", "Cell Width","Cell Height"]].to_numpy()
        data_labels = pd.qcut(df.loc[:,"Line1_Combined"],num_classes, labels=False).to_numpy()
    wvltnp = np.vstack(wvlt.apply(convert_col_to_np).to_numpy())
    # wvltnp2 = np.vstack(wvlt2.apply(convert_col_to_np).to_numpy())
    # wvltnp3 = np.vstack(wvlt3.apply(convert_col_to_np).to_numpy())
    # all_feats = np.concatenate((others,wvltnp,wvltnp2,wvltnp3),axis=1)
    all_feats = np.concatenate((others,wvltnp),axis=1)
    data_labels = keras.utils.to_categorical(data_labels, num_classes)
    return train_test_split(all_feats, data_labels, test_size=TEST_FRACTION, random_state=seed)

def get_training_data(num_classes):
    def convert_col_to_np(str_array):
        #f2re = np.fromstring(f2str.strip('[]'), sep= ' ', dtype = np.float32)
        # IPython.embed()
        return list(np.fromstring(str_array.strip('[]'), sep= ' ', dtype = np.float32))
    df = pd.read_csv(path_to_data)
    # IPython.embed()
    wvlt = df.loc[df["Cancer?"]=="Cancer", "Wavelet Scattering Vector One"]
    # wvlt = df.loc[:, "Wavelet Scattering Vector One"]
    wvltnp = np.vstack(wvlt.apply(convert_col_to_np).to_numpy())

    data_labels = pd.qcut(df.loc[df["Cancer?"]=="Cancer","Line1_Combined"],num_classes, labels=False).to_numpy()
    # data_labels = pd.qcut(df.loc[:,"Line1_Combined"],num_classes, labels=False).to_numpy()

    data_labels = keras.utils.to_categorical(data_labels, num_classes)
    # a,b,c,d = train_test_split(wvltnp, data_labels, test_size=0.5, random_state=42)
    # IPython.embed()
    return train_test_split(wvltnp, data_labels, test_size=TEST_FRACTION, random_state=seed)

def create_network(x_train, num_classes):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape = x_train[0].shape)) # (30,) or (81,)
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics = ['accuracy'])
    return model

def main():

    batch_size = 128
    epochs = 40
    num_classes = 2
    x_train, x_test, y_train, y_test = get_training_data_plus(num_classes)
    model = create_network(x_train, num_classes)
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs,verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n Test Loss: {score[0]}\nTest Accuracy: {score[1]}")


if __name__=='__main__':
    main()