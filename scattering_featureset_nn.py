'''
Neural network to classify LINE1 content based on features generated from cells in the c4 CosMx run
Two different kinds of features: texture features, and geometric.

'''

import warnings
warnings.simplefilter('ignore') # doesn't work
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # this knocks out the tensorflow debugging information
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import IPython
import math
from copy import copy


# path_to_data = r"../wavelet_scattering_J2.csv"
# path_to_data = r"../3.15.23_synthetic_b=10_o=5_lir=15_NOBLANKS.csv"
# path_to_data = r"../3.15.23_padded_J=2.csv"
# path_to_data = r"../3.21.23_padded_geometric_features_only.csv"
# path_to_data = r"../3.21.23_scattering_plus_geomtric_padded_J=2.csv"
seed = 55
TEST_FRACTION = 0.1
# GEOM_FEATS = ["Perimeter","Bounding ellipse perimeter","Extent", "Solidity", "Eccentricity","Euler number"]
# GEOM_FEATS = ["Feret diameter min","Bounding ellipse perimeter","Minor axis","Euler number"]
# GEOM_FEATS = ["Feret diameter max", "Feret diameter min"]
# GEOM_FEATS = ["Entire cell area","Nuclear area convex", "Nuclear area filled","Major axis","Minor axis", # Geometric feats
#                                  "Bounding ellipse area","Bounding ellipse perimeter", "Eccentricity","Feret diameter max",# Geometric feats
#                                 "Feret diameter min","Perimeter","Solidity", "Extent", "Euler number", #Geometric feats
# #                                 "DAPI Intensity Mean"]

GEOM_FEATS = ["Entire cell area","Nuclear area convex", "Nuclear area filled","Major axis","Minor axis", # Geometric feats
                                 "Bounding ellipse area","Bounding ellipse perimeter", "Eccentricity","Feret diameter max",# Geometric feats
                                "Feret diameter min","Perimeter","Solidity", "Extent", "Euler number", #Geometric feats
                                "DAPI Intensity Mean", 'esf','csf','sf1','sf2','elongation']
GLCM_FEATS = ["Texture-correlation","Texture-dissimilarity","Texture-homogeneity","Texture-ASM","Texture-energy","Texture-contrast"]

def compile_featureset(choices, wvlt, hu, whu, geom, glcm):
    first = choices.pop().lower()
    match first:
        case "wavelet scattering":
            all_feats = wvlt
        case "hu":
            all_feats = hu
        case "weighted hu":
            all_feats = whu
        case "geometric":
            all_feats = geom
        case "glcm":
            all_feats = glcm
        case _:
            raise Exception(f"Unexpected choice of feature to train on: {first}")
    while len(choices) > 0:
        next = choices.pop()
        match next:
            case "wavelet scattering":
                all_feats = np.concatenate((all_feats,wvlt),axis=1)
            case "hu":
                all_feats = np.concatenate((all_feats,hu),axis=1)
            case "weighted hu":
                all_feats = np.concatenate((all_feats,whu),axis=1)
            case "geometric":
                all_feats = np.concatenate((all_feats,geom),axis=1)
            case "glcm":
                all_feats = np.concatenate((all_feats,glcm),axis=1)
            case _:
                raise Exception(f"Unexpected choice of feature to train on: {next}")
    return all_feats

def squish_to_range(arr, method = "zero to one"):
    modified = copy(arr)
    def squish(col):
        col_std = np.std(col)
        col_mean = np.mean(col)
        col_min = np.min(col)  
        col_max = np.max(col)
        def f(x):
            return (x - col_mean) / col_std
        if method == "zero to one":  
            return np.interp(col,[col_min,col_max],[0,1])
        elif method == "z score":
            return np.vectorize(f)(col)
        else:
            raise Exception(f"Unsupported method passed to squish fn: {method}")
    return np.apply_along_axis(squish,0,modified)

def add_secondary_geometrics(df):
    df["esf"] = df['Minor axis'] / df['Major axis']
    df['csf'] = (4*math.pi* df['Bounding ellipse area']) / df['Bounding ellipse perimeter']**2
    df['sf1'] = df['Minor axis'] / df['Feret diameter max']
    df['sf2'] = df['Feret diameter min'] / df['Feret diameter max']
    df['elongation'] = df['Feret diameter max'] / df['Feret diameter min']
    # df.assign(convexity=lambda df: math.sqrt(df['Bounding ellipse area']/df['Nuclear area convex'])) #= math.sqrt(df['Bounding ellipse area']/df['Nuclear area convex'])    
    return df

def get_all_training_data(df, num_classes, subset_to_cancer = True,train_on = "Line1", featureset = ['wavelet scattering'], normalize_by = None ):
    def convert_hu_to_np(str_array):
        hu_original = list(np.fromstring(str_array.strip('[]'), sep= ' ', dtype = np.float32))
        hu_scaled = []
        for i in range(0,7):
            moment = hu_original[i]
            if moment == 0:
                hu_scaled.append(0)
            else:
                hu_scaled.append(-1 * math.copysign(1.0,moment) * math.log10(abs(moment)))
        hu_scaled[-1] = abs(hu_scaled[-1])
        return hu_scaled
    def convert_wvlt_to_np(str_array):
        return list(np.fromstring(str_array.strip('[]'), sep= ' ', dtype = np.float32))[1:] # remove first (just lowpass)
    # df = pd.read_csv(path_to_data).dropna() # Seems that there are some holes in the data, not sure why... need to investigate
    df = add_secondary_geometrics(df)
    if subset_to_cancer:
        wvlt = df.loc[df["Cancer?"]=="Cancer", "Wavelet Scattering Vector One"]
        hu = df.loc[df["Cancer?"]=="Cancer", "Hu moments"]
        whu = df.loc[df["Cancer?"]=="Cancer", "Weighted Hu moments"]
        geom = df.loc[df["Cancer?"]=="Cancer", GEOM_FEATS].to_numpy()
        glcm = df.loc[df["Cancer?"]=="Cancer", GLCM_FEATS].to_numpy()
        data_labels = pd.qcut(df.loc[df["Cancer?"]=="Cancer","Line1_Combined"],num_classes, labels=False).to_numpy()
    else:
        wvlt = df.loc[:, "Wavelet Scattering Vector One"]
        hu = df.loc[:, "Hu moments"]
        whu = df.loc[:, "Weighted Hu moments"]
        geom = df.loc[:, GEOM_FEATS].to_numpy()
        glcm = df.loc[:, GLCM_FEATS].to_numpy()
        if train_on.lower() == 'line1':
            data_labels = pd.qcut(df.loc[:,"Line1_Combined"],num_classes, labels=False).to_numpy()
        elif train_on.lower() == 'cancer':
            data_labels = np.where(df['Cancer?'] == 'Cancer', 1,0)
        else:
            raise Exception("Can't train on this column")
    wvltnp = np.vstack(wvlt.apply(convert_wvlt_to_np).to_numpy())
    hunp = np.vstack(hu.apply(convert_hu_to_np).to_numpy())
    whunp = np.vstack(whu.apply(convert_hu_to_np).to_numpy())
    all_feats = compile_featureset(copy(featureset), wvlt=wvltnp, hu=hunp, whu=whunp, geom=geom, glcm=glcm)

    if normalize_by is not None:
        all_feats = squish_to_range(all_feats, normalize_by)
    # IPython.embed()
    data_labels = keras.utils.to_categorical(data_labels, num_classes)
    print(f'\nTraining on {featureset}\n with {all_feats.shape[1]} total parameters')
    return train_test_split(all_feats, data_labels, test_size=TEST_FRACTION, random_state=seed)

def get_geometric_training_data(num_classes, cancer = "Cancer"):
    cols_to_train_on = ["Entire cell area","Nuclear area convex", "Nuclear area filled","Major axis","Minor axis", # Geometric feats
                                 "Bounding ellipse area","Bounding ellipse perimeter", "Eccentricity","Feret diameter max",# Geometric feats
                                "Feret diameter min","Perimeter","Solidity", "Extent", "Euler number", #Geometric feats
                                "DAPI Intensity Mean"]
    def convert_col_to_np(str_array):
        return list(np.fromstring(str_array.strip('[]'), sep= ' ', dtype = np.float32))
    df = pd.read_csv(path_to_data).dropna() # Seems that there are some holes in the data, not sure why... need to investigate
    if cancer:
        hu = df.loc[df["Cancer?"]==cancer, "Hu moments"]
        whu = df.loc[df["Cancer?"]==cancer, "Weighted Hu moments"]
        others = df.loc[df["Cancer?"]==cancer, cols_to_train_on].to_numpy()
        data_labels = pd.qcut(df.loc[df["Cancer?"]==cancer,"Line1_Combined"],num_classes, labels=False).to_numpy()
    else:
        hu = df.loc[:, "Hu moments"]
        whu = df.loc[:, "Weighted Hu moments"]
        others = df.loc[:, cols_to_train_on].to_numpy()
        data_labels = pd.qcut(df.loc[:,"Line1_Combined"],num_classes, labels=False).to_numpy()
    hunp = np.vstack(hu.apply(convert_col_to_np).to_numpy())
    whunp = np.vstack(whu.apply(convert_col_to_np).to_numpy())
    all_feats = np.concatenate((others,hunp,whunp),axis=1)
    data_labels = keras.utils.to_categorical(data_labels, num_classes)
    return train_test_split(all_feats, data_labels, test_size=TEST_FRACTION, random_state=seed)

def get_scattering_training_data_plus(num_classes, cancer = "Cancer"):
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

def get_scattering_training_data(num_classes):
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

def create_dense_network(x_train, num_classes):
    model = Sequential()
    model.add(Dense(25, activation='relu', input_shape = x_train[0].shape)) # (30,) or (81,)
    # model.add(Dense(20, activation='relu')) # (30,) or (81,)
    # model.add(Dense(15, activation='relu')) # (30,) or (81,)
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss = 'categorical_crossentropy',optimizer=opt, metrics = ['accuracy'])
    return model

def create_simple_network(x_train, num_classes):
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', input_shape = x_train[0].shape)) 
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.003)
    model.compile(loss = 'categorical_crossentropy',optimizer=opt, metrics = ['accuracy'])
    return model

def create_ordinal_network(x_train, num_classes):
    import coral_ordinal as coral
    model = Sequential()
    model.add(Dense(25, activation='relu', input_shape = x_train[0].shape)) # (30,) or (81,)
    # model.add(Dense(20, activation='relu')) # (30,) or (81,)
    # model.add(Dense(15, activation='relu')) # (30,) or (81,)
    # model.add(Dropout(0.2))
    model.add(coral.CoralOrdinal(num_classes))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss = coral.OrdinalCrossEntropy(),optimizer=opt, metrics = ['accuracy', coral.MeanAbsoluteErrorLabels()])
    return model

def examine_simple_nn(weights, cols):
    low = weights[:,0]
    high = weights[:,1]
    low_results = {}
    high_results = {}
    for i,col in enumerate(cols):
        low_results[col] = low[i]
        high_results[col] = high[i]
    return low_results, high_results

def predict_fov_cancer_line1_content(path_to_test_data, models, normalize_by = None ):
    # Expects an array with run ID and fov, an array with features to predict on, and a model 
    def convert_hu_to_np(str_array):
        hu_original = list(np.fromstring(str_array.strip('[]'), sep= ' ', dtype = np.float32))
        hu_scaled = []
        for i in range(0,7):
            moment = hu_original[i]
            if moment == 0:
                hu_scaled.append(0)
            else:
                hu_scaled.append(-1 * math.copysign(1.0,moment) * math.log10(abs(moment)))
        hu_scaled[-1] = abs(hu_scaled[-1])
        return hu_scaled
    def convert_wvlt_to_np(str_array):
        return list(np.fromstring(str_array.strip('[]'), sep= ' ', dtype = np.float32))[1:]
    df = pd.read_csv(path_to_test_data).dropna()
    # IPython.embed()
    df = add_secondary_geometrics(df)
    descriptors = df.loc[df["Cancer?"]=="Cancer", ["fov"]]
    wvlt = df.loc[df["Cancer?"]=="Cancer", "Wavelet Scattering Vector One"]
    wvltnp = np.vstack(wvlt.apply(convert_wvlt_to_np).to_numpy())
    whu = df.loc[df["Cancer?"]=="Cancer", "Weighted Hu moments"]
    whunp = np.vstack(whu.apply(convert_hu_to_np).to_numpy())
    geom = df.loc[df["Cancer?"]=="Cancer", GEOM_FEATS].to_numpy()
    glcm = df.loc[df["Cancer?"]=="Cancer", GLCM_FEATS].to_numpy()
    if normalize_by is not None:
        wvltnp = squish_to_range(wvltnp, normalize_by)
        whunp = squish_to_range(whunp, normalize_by)
        glcm = squish_to_range(glcm, normalize_by)
        geom = squish_to_range(geom, normalize_by)


    decile_stats = pd.read_csv(r"/home/peter/home_projects/CosMx/D10_cancerOnly_Line1_decile_stats.csv")
    def get_col_mean(one_hot_row):
        class_num = np.argmax(one_hot_row)
        if class_num == 0 or class_num == '0':
            start = str(class_num)
        else:
            start = str(class_num) + '0'
        end = str(class_num+1)+'0'
        col = start + ' to '+ end
        val = decile_stats.loc[decile_stats['L1_decile'] == col, "Line1_Combined mean"].iloc[0]
        return int(val)
    wavelet_preds = models[0].predict(wvltnp)
    geom_preds = models[1].predict(geom)
    glcm_preds = models[2].predict(glcm)
    whu_preds = models[3].predict(whunp)
    a = np.random.randint(9,size=wavelet_preds.shape[0])
    random_preds = keras.utils.to_categorical(a, 10)
    # IPython.embed()
    pred_values = np.column_stack((np.apply_along_axis(get_col_mean,1,wavelet_preds),
                            np.apply_along_axis(get_col_mean,1,geom_preds),
                            np.apply_along_axis(get_col_mean,1,glcm_preds),
                            np.apply_along_axis(get_col_mean,1,whu_preds),
                            np.apply_along_axis(get_col_mean,1,random_preds)))
    # IPython.embed()
    # pred_values = np.apply_along_axis(get_col_mean,1,preds)
    predicted_line1_counts = pd.DataFrame(pred_values, columns = ['Wavelet_Predicted_Line1','Geometric_Predicted_Line1','GLCM_Predicted_Line1','WeightedHu_Predicted_Line1','Randomly_Predicted_Line1'])
    return pd.concat([descriptors.reset_index(drop=True),predicted_line1_counts.reset_index(drop=True)], axis = 1)

def combine_runs(list_of_runs):
    # Should be a tuple with (path_to_run, str_label)
    path1, run_name = list_of_runs.pop()
    df = pd.read_csv(path1).dropna()
    df["Run"] = run_name
    while list_of_runs != []:
        path2, run_name2 = list_of_runs.pop()
        df2 = pd.read_csv(path2).dropna()
        df2["Run"] = run_name2
        df = pd.concat([df2,df])
    return df

def reverse_categorical(x):
    return np.argmax(x)

def main():
    trainC4 = r"../3.29.23_paddedInput200_C4_1&5glcm_J2wavelet_allGeom.csv"
    trainB10 =  r"../3.29.23_paddedInput200_B10_1&5glcm_J2wavelet_allGeom.csv"
    trainD10 = r"../3.29.23_paddedInput200_D10_1&5glcm_J2wavelet_allGeom.csv"
    raw_train = combine_runs([(trainD10,"D10")])
    # path_to_testSet = r"../3.28.23_rawInput_B10_1&5glcm_J2wavelet_allGeom.csv"
    path_to_testSet = trainC4 # C4
    batch_size = 128
    epochs = 100
    num_classes = 10
    featureset = [['wavelet scattering'], ['geometric'], ['glcm'],['weighted hu']] # 'glcm' 'geometric' 'wavelet scattering' 'hu' 'weighted hu'
    cancer = True # True or False
    dv = 'line1' # 'line1' or 'cancer'
    norm = None # 'zero to one' or 'z score' or None
    models = []
    while featureset != []:
        features = featureset[0]
        featureset.remove(features)

        print(f'\nPre-processing input data... ', end='')
        x_train, x_test, y_train, y_test = get_all_training_data(raw_train,num_classes,subset_to_cancer = cancer,train_on = dv,featureset = features, normalize_by=norm)
        print("Done")
        # model = create_simple_network(x_train, num_classes)
        # model = create_dense_network(x_train, num_classes)
        model = create_ordinal_network(x_train, num_classes)
        y_train = np.apply_along_axis(reverse_categorical,1,y_train) # CORAL takes one number
        y_test = np.apply_along_axis(reverse_categorical,1,y_test)
        history = model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs,verbose=1, validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n Test Loss: {round(score[0],5)}\nTest Accuracy: {round(score[1], 5) *100}%")
        models.append(model)
        if cancer:
            population = 'cancer cells only'
        else:
            population = 'all cell types'
        if dv.lower() == 'line1':
            print(f'\nThis model was trained to predict Line1 content to the nearest {int(100/num_classes)}% in {population} using the follow features: {", ".join(x for x in features)}')
        elif dv.lower() == 'cancer':
            print(f'\nThis model was trained to recognize cancer in {population} using the following features: {", ".join(x for x in features)}')

    predictions = predict_fov_cancer_line1_content(path_to_testSet, models)
    fov_pred = predictions.groupby(by="fov").sum()
    # IPython.embed()
    predictions.to_csv(r"../C4_cellPerRow_ORDINALpredictions_100epochsD10_allPadded.csv", index = False)
    fov_pred.to_csv(r"../C4_fov_ORDINALpredictions_100epochsD10_allPadded.csv")
    # inputWeights = model.weights[0].numpy()
    # l, h = examine_simple_nn(inputWeights, GLCM_FEATS)
    # ls = list(reversed(sorted(l.items(), key=lambda x:abs(x[1]))))
    # hs = list(reversed(sorted(h.items(), key=lambda x:abs(x[1]))))
    # IPython.embed()

if __name__=='__main__':
    main()