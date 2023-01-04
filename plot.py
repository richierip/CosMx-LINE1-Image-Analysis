'''
Compare nuclear intensity vs LINE1 transcripts per cell, run correlation / regression / whatever else
    This file takes in a curated data table and makes a plot
Peter Richieri 
MGH Ting Lab 
11/9/22
'''

import seaborn as sns
from seaborn import lmplot
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys

custom_data_path = r"..\CosMx_C4_CellResults_GaborFeats2.csv"

def bin_by_dapi(df, norm_by_area=False):
    def _calc_norm_area(row):
        return row["DAPI Intensity Mean"] / row["DAPI Area (px)"]
    # def _calc_percentile(row, colmax):
    #     return row["DAPI Intensity over Area"] *100 / colmax
    def _label_decile(row):
        rank = row["DAPI Intensity over Area Decile"]
        if rank == 0 or rank == '0':
            start = str(rank)
        else:
            start = str(rank) + '0'
        end = str(rank+1)+'0'
        return start + ' to '+ end
    if norm_by_area:
        # independent_var = "DAPI Intensity over Area"
        bin_name = "DAPI_Area_Norm_Bin"
        df["DAPI Intensity over Area"] = df.apply(lambda row:_calc_norm_area(row), axis=1)
        df["DAPI Intensity over Area Decile"] = pd.qcut(df["DAPI Intensity over Area"],10, labels=False)
        #df.apply(lambda row:_calc_percentile(row,df.loc[df['DAPI Intensity over Area'].idxmax()]['DAPI Intensity over Area']), axis=1)
        # independent_var = "DAPI Intensity over Area Percentile"
        df["DAPI_Area_Norm_Bin"] = df.apply(lambda row: _label_decile(row),axis=1)

    elif not norm_by_area:
        independent_var = "DAPI Intensity Mean"
        bin_name = "DAPI_Bin"
    
    # print(f"\nmy iv is {independent_var} and bin name is {bin_name}")
        df.insert(4,bin_name,"")
        df.loc[df[independent_var] <15, bin_name] = "0 to 15"
        df.loc[(15< df[independent_var]) & (df[independent_var] <=20), bin_name] = "15 to 20"
        df.loc[(20< df[independent_var]) & (df[independent_var] <=30), bin_name] = "20 to 30"
        df.loc[(30<df[independent_var]) & (df[independent_var] <=40), bin_name] = "30 to 40"
        df.loc[(40<df[independent_var]) & (df[independent_var] <=50), bin_name] = "40 to 50"
        df.loc[(50<df[independent_var]) & (df[independent_var] <=60), bin_name] = "50 to 60"
        df.loc[(60<df[independent_var]) & (df[independent_var] <=70), bin_name] = "60 to 70"
        df.loc[(70<df[independent_var]) & (df[independent_var] <=80), bin_name] = "70 to 80"
        df.loc[(80<df[independent_var]) & (df[independent_var] <=90), bin_name] = "80 to 90"
        df.loc[(90<df[independent_var]) & (df[independent_var] <=101), bin_name] = "90 to 100"
        if independent_var == "DAPI Intensity Mean":
            df.loc[df[independent_var] >101, bin_name] = ">100"
    df.to_csv(custom_data_path, index=False)
    return df

def make_column(row):
    if row['Cell type'] in ['CAF','macrophage','cancer','T.cell','Vascular.smooth.muscle']:
        return row['Cell type']
    else:
        return 'Other'

def scatter(path_to_data, cmd_args):
    model = cmd_args[1]
    if len(cmd_args)>2:
        iv = cmd_args[2]
        if iv == 'intensity':
            independent_variable = "DAPI Intensity Mean"
        elif iv == 'area':
            independent_variable = "DAPI Intensity over Area"
        else:
            print("\nNeed 'intensity' or 'area'")
            exit(0)
    def annotate(data, **kws):
        if model == "linear":
            r, p = stats.pearsonr(data[independent_variable], data["Line1_Combined"])
            ax = plt.gca()
            ax.text(.5, .8, 'Pearson r={:.2f}, p={:.2g}'.format(r, p),transform=ax.transAxes)
        elif model == "polynomial":
            r, p = stats.spearmanr(data[independent_variable], data["Line1_Combined"])
            ax = plt.gca()
            ax.text(.5, .8, 'Spearman r={:.2f}, p={}'.format(r, p),transform=ax.transAxes)
        else:
            print("\nNeed 'linear' or 'polynomial'")
            exit(0)
            
    df = pd.read_csv(path_to_data).dropna() # remove N/As for scipy regression stats
    # cancer_only = df[df["Cancer?"]=="Not Cancer"]
    df["Interesting cell types"] = df.apply(lambda row:make_column(row), axis=1)
    if model == "linear":
        p = lmplot(data=df, x=independent_variable, y="Line1_Combined",
                    col="Cancer?",  scatter_kws={"s":2})
    elif model == "polynomial":
        p = lmplot(data=df, x=independent_variable, y="Line1_Combined",
                    col="Cancer?", order=3, scatter_kws={"s":2})
    # sns.lmplot(data=cancer_only, x="DAPI Intensity Mean", y="Line1_Combined")
    elif model =='scatter':
        p = sns.relplot(data=df, x=independent_variable, y="Line1_Combined",
                    col="Cancer?", hue = 'Interesting cell types',alpha = 0.1,  size=1.2, edgecolor=None)
    else:
        print("Bad input")
        exit(0)
    if model != 'scatter':
        p.map_dataframe(annotate)
    plt.show()

def plot_dapi_bins(pre_process_df, cmd_args):
    if len(cmd_args) >2:
        bar_type = cmd_args[2]
    if len(cmd_args)>3:
        if cmd_args[3]== 'intensity':
            independent_var = "DAPI_Bin"
            title_var = ' DAPI Intensity (binned)'
        elif cmd_args[3]=='area':
            independent_var = "DAPI_Area_Norm_Bin"
            title_var = " DAPI Intensity over Area (percentile rank)"
    else:
        bar_type = "mean"
        independent_var = "DAPI_Bin"
        title_var = ' DAPI Intensity (binned)'

    # df = bin_by_dapi(pre_process_df)
    df = pre_process_df
    if independent_var == "DAPI_Bin":
        x_order = ["0 to 15","15 to 20","20 to 30","30 to 40","40 to 50","50 to 60","60 to 70","70 to 80","80 to 90","90 to 100",">100"]
    elif independent_var == "DAPI_Area_Norm_Bin":
        x_order = ["0 to 10","10 to 20","20 to 30","30 to 40","40 to 50","50 to 60","60 to 70","70 to 80","80 to 90","90 to 100"]
        
    
    l1bins = sns.barplot(data=df, x=independent_var,y="Line1_Combined", errorbar='sd',
                estimator = bar_type,order = x_order, hue="Cancer?").set(title= bar_type.title()+title_var+' vs ORF1/2 Expression')
    # axis_flip = sns.barplot(data=df, y=independent_var,x="Line1_Combined", errorbar='sd',
    #             estimator = bar_type,order = x_order, hue="Cancer?").set(title= 'L1 ORF1/2 Expression vs '+bar_type.title()+title_var)
    plt.show()
    l1counts = sns.countplot(data=df, x=independent_var,hue="Cancer?",order=x_order).set(title= 'Cell Counts by '+title_var)
    
    # print(f"type of counts is {type(l1counts)} and bins is {type(l1bins)}")
    
    # l1counts.bar_label(l1counts.containers[0])
    plt.show()

def l1_vs_counts(datapath,cmd_args):

    df = pd.read_csv(datapath).dropna()
    df["Interesting cell types"] = df.apply(lambda row:make_column(row), axis=1)
    p = sns.relplot(data=df, x='Total transcript counts', y="Line1_Combined",
                     col="Interesting cell types",hue='DAPI Intensity over Area Decile',
                       col_wrap=4, size=1.2, alpha=0.2,edgecolor=None)
    plt.show()

def l1_vs_texture(datapath,cmd_args):
    if len(cmd_args) > 3:
        texture_option = cmd_args[3]
    else:
        texture_option = 'correlation'
    df = pd.read_csv(datapath).dropna()
    df["Interesting cell types"] = df.apply(lambda row:make_column(row), axis=1)
    p = sns.relplot(data=df, x=f'Texture-{texture_option}', y="Line1_Combined",
                     col="Interesting cell types",col_wrap=3, hue = "Interesting cell types",alpha=0.3, size=0.9,edgecolor=None)
    plt.show()
    
def plot_texture_bars(datapath,cmd_args):
    df = pd.read_csv(datapath).dropna()
    df["Interesting cell types"] = df.apply(lambda row:make_column(row), axis=1)

    if len(cmd_args) > 3:
        texture_option = cmd_args[3]
    else:
        texture_option = 'correlation'

    if texture_option in ['correlation','ASM','energy','dissimilarity','contrast','homogeneity']:
        texture_option = f'Texture-{texture_option}'
    else:
        freq = cmd_args[3]
        s = cmd_args[4]
        texture_option = f'Gabor {freq} {s}'

    bar_type = 'mean'

    l1bins = sns.barplot(data=df, x='Interesting cell types',y=texture_option, errorbar='se',
                estimator = bar_type, hue="Cancer?").set(title= f'{bar_type.title()} {texture_option} by cell type')
    plt.show()

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'linear' or sys.argv[1] == 'polynomial'or sys.argv[1] == 'scatter':
            if sys.argv[1] != 'scatter':
                print("Preparing to plot a trendline...")
            else: 
                if sys.argv[2] == 'texture':
                    l1_vs_texture(custom_data_path, sys.argv)
                    exit(0)
                print("Assembling scatterplots...")
                if len(sys.argv) > 2 and sys.argv[2] == 'counts':
                    l1_vs_counts(custom_data_path,sys.argv)
                    exit(0)

            scatter(custom_data_path, sys.argv)
        elif sys.argv[1] == 'barplot':
            print("Assembling bar plots ...")
            if sys.argv[2] == 'texture':
                plot_texture_bars(custom_data_path,sys.argv)
            else:
                plot_dapi_bins(pd.read_csv(custom_data_path).dropna(), sys.argv)
        else:
            print("Check your input.")
    else:
        print("\nCalculating bins to append to input dataframe...   ",end='')
        bin_by_dapi(df = pd.read_csv(custom_data_path).dropna(), norm_by_area=True)
        print("Done.")

if __name__=='__main__':
    main()