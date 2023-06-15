import pandas as pd
import numpy as np
import IPython


def _label_decile(row):
    try:
        rank = int(row["L1_decile_raw"])
    except:
        # pass
        IPython.embed()
    if rank == 0 or rank == '0':
        start = str(rank)
    else:
        start = str(rank) + '0'
    end = str(rank+1)+'0'
    return start + ' to '+ end

def _label_high_low(row):
    try:
        x = int(row['L1_binary_raw'])
    except:
        print("something went wrong")
        IPython.embed()

    if x == 0:
        return "Low"
    elif x ==1:
        return "High"
    else:
        raise Exception("Not binary")

def combine_runs(list_of_runs):
    # Should be a tuple with (path_to_run, str_label)
    df, run_name = list_of_runs.pop()
    df["Run"] = run_name
    while list_of_runs != []:
        df2, run_name2 = list_of_runs.pop()
        df2["Run"] = run_name2
        df = pd.concat([df2,df])
    return df

def get_l1_by_fov(df,pathname, run_type = "all"):
    IPython.embed()
    if run_type == "single":
        sum_per_fov = df.loc[df["Cancer?"]=="Cancer",("fov","Line1_Combined")].groupby(by="fov").sum()
        sum_per_fov.to_csv(pathname)
    elif run_type=='all':
        sum_per_fov = df.loc[df["Cancer?"]=="Cancer",("Run name", "fov","Line1_combined")].groupby(by=["Run name","fov"]).sum()
        return None


def get_mean_l1_per_decile(df, pathname, num_classes=10):
    df["L1_decile_raw"] = pd.qcut(df.loc[df["Cancer?"]=="Cancer","Line1_Combined"],num_classes, labels=False)
    df["L1_decile"] = df.loc[df["Cancer?"]=="Cancer"].apply(lambda row: _label_decile(row),axis=1)
    stats_per_decile = df.loc[df["Cancer?"]=="Cancer", ["L1_decile","Line1_Combined"]].groupby(by="L1_decile").agg(['sum','mean'])
    
    # Convert frame from MultiIndex back to normal column headers 
    stats_per_decile.columns = list(map(' '.join, stats_per_decile.columns.values))

    #D10
    stats_per_decile.to_csv(pathname)

def glcm_scatter(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    dfc = df.loc[df["Cancer?"]=="Cancer"].copy()
    # g= sns.FacetGrid(data=df)
    # g.map(plt.scatter,"Texture-dissimilarity","Texture-correlation")
    # g.map(sns.regplot,"Texture-dissimilarity","Texture-correlation") 
    p = sns.scatterplot(data=df, x="Texture-dissimilarity", y="Texture-correlation")
    p = sns.scatterplot(data=dfc, x="Texture-dissimilarity", y="Texture-correlation")

    plt.show()

    dfc["L1_binary_raw"] = pd.qcut(dfc.loc[:,"Line1_Combined"],2, labels=False)
    dfc["L1_binary"] = dfc.apply(lambda row: _label_high_low(row),axis=1)

    p = sns.scatterplot(data=dfc, x="Texture-dissimilarity", y="Texture-correlation", 
                        hue = "L1_binary", palette = {"High":"red", "Low":"green"}, alpha = 0.5)
    
    plt.show()
    # IPython.embed()
    exit()

def main():
    c4 = r"../3.29.23_paddedInput200_C4_1&5glcm_J2wavelet_allGeom.csv"
    b10 =  r"../3.29.23_paddedInput200_B10_1&5glcm_J2wavelet_allGeom.csv"
    d10 = r"../3.29.23_paddedInput200_D10_1&5glcm_J2wavelet_allGeom.csv"
    all = r"../4.27.23_allRuns_fromScratch_fixedCompartmentTX_erosion_GLCM+geom.csv"
    # c4df =  pd.read_csv(c4).dropna()
    # b10df = pd.read_csv(b10).dropna()
    # d10df = pd.read_csv(d10).dropna()
    # IPython.embed()
    # get_l1_by_fov(c4df, r"../C4_cancerOnly_Line1_sum_byFOV.csv")
    # get_l1_by_fov(b10df, r"../B10_cancerOnly_Line1_sum_byFOV.csv")
    # glcm_scatter(c4df)

    get_l1_by_fov(all, r"../allRuns_cancerOnly_Line1_sum_byFOV.csv")

    # b10d10 = combine_runs([(b10df, "B10"),(d10df,"D10")]).reset_index(drop=True)
    # get_mean_l1_per_decile(b10d10, r"../B10D10_cancerOnly_Line1_decile_stats.csv")

    # c4d10 = combine_runs([(c4df, "C4"),(d10df,"D10")]).reset_index(drop=True)
    # get_mean_l1_per_decile(c4d10, r"../C4D10_cancerOnly_Line1_decile_stats.csv")

    # c4b10 = combine_runs([(c4df, "C4"),(b10df,"B10")]).reset_index(drop=True)
    # get_mean_l1_per_decile(c4b10, r"../C4B10_cancerOnly_Line1_decile_stats.csv")

if __name__=='__main__':
    main()