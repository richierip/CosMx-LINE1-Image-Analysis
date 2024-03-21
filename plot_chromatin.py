import numpy as np
import pandas as pd
import seaborn as sns
import IPython
# from remote_plot import plt

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# path = r"/home/peter/home_projects/CosMx/4.27.23_allRuns_fromScratch_fixedCompartmentTX_erosion_GLCM+geom.csv"
path = r"/home/peter/home_projects/CosMx/5.1.23_allRuns_fromScratch_CAFerosion_GLCM+geom.csv"
df = pd.read_csv(path)

# Drop junk rows (data from txs that have no cell underneath)
df = df[df['local_cid'] !=0]
# make various ratio cols
df['Line1_combined_nuclear_log10'] = np.log10(df['Line1_combined_nuclear']+1)
df['Line1_combined_cytoplasm_log10'] = np.log10(df['Line1_combined_cytoplasm']+1)
df['L1_nuc_copy'] = df["Line1_combined_nuclear"] 
df['L1_nuc_copy'] = np.where(df['L1_nuc_copy'] == 0, 0.5,df["L1_nuc_copy"])
df['L1_cyt_copy'] = df["Line1_combined_cytoplasm"] 
df['L1_cyt_copy'] = np.where(df['L1_cyt_copy'] == 0, 0.5,df["L1_cyt_copy"])
df['L1_compartment_ratio'] = df["Line1_combined_nuclear"] /df["L1_cyt_copy"] 
df['L1_compartment_difference'] = df["Line1_combined_nuclear"] - df["Line1_combined_cytoplasm"] 
df['L1_compartment_dif_ofTotal'] = df['L1_compartment_difference'] * 100 / df['Line1_combined'] 

df_chroma = df.loc[df['Inner nucleus intensity mean'] != "Too small"]
df_chroma = df_chroma.astype({'Inner nucleus intensity mean': 'float', 'Outer nucleus intensity mean':float})
df_chroma['Chromatin_intensity_ratio'] = df_chroma['Outer nucleus intensity mean'] / df_chroma['Inner nucleus intensity mean']


# Drop zeros in these columns so the ratio is never infinite. 
df_chroma = df_chroma[df_chroma['Outer nucleus intensity mean'] != 0]
df_chroma = df_chroma[df_chroma['Inner nucleus intensity mean'] != 0]
df_chroma_copy = df_chroma.copy()

facet_vars = ["Line1_combined_nuclear", "Line1_combined_cytoplasm",  "Line1_combined", "Mean PanCK", "Mean CD45", 
              "Mean CD3", "Full nucleus intensity mean", "Full nucleus area (px)", 'Chromatin_intensity_ratio']
df_chroma_cancer = df_chroma_copy.loc[df_chroma_copy['Cell type'] == "cancer"]
df_cancer2 = df_chroma_copy.loc[df_chroma_copy['Cancer?'] != "No data"]
df_chroma_nd = df_chroma[df_chroma["Cancer?"] != "No data"]

df_chroma['Outer/inner_ratio_qcut']= pd.qcut(df_chroma['Chromatin_intensity_ratio'], 15)
bins = pd.IntervalIndex.from_tuples([(.2, .3), (.3, .4), (.4, .5), (.5,.6), (.6,.7),(.7,.8),
                                     (.8,.9),(.9,1),(1,1.1),(1.1,1.2),(1.2,1.3),(1.3,1.4),(1.4,1.5),(1.5,2.5)])
histbins = [.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,2.5]
df_chroma['Outer/inner_ratio_cut']= pd.cut(df_chroma['Chromatin_intensity_ratio'], bins)
df_chroma['Line1_combined_qcut']= pd.qcut(df_chroma['Line1_combined'], 10)
df['Line1_combined_qcut']= pd.qcut(df['Line1_combined'], 10)


#---------------------- CAF area Histograms
# p = sns.histplot(data = df[df["Cell type"] == "CAF"], x = 'Entire cell area').set(title = "Histogram of CAF entire cell area")
# plt.show()
# cafs = df[df["Cell type"] == "CAF"].copy()
# cafs = cafs[cafs["Full nucleus area (px)"] !="Too small"]
# cafs = cafs.astype({'Full nucleus area (px)': 'float'})
# # IPython.embed()
# q = sns.histplot(data = cafs, x = 'Full nucleus area (px)', bins = 500).set(title = "Histogram of CAF nuclear area")
# plt.show()


#---------------------------------------------------------
#-------------------------------Line1 compartment jointplot
# df.dropna(subset = ['Full nucleus intensity mean'], inplace=True)
# df = df.loc[df['Full nucleus intensity mean'] !="Too small"]
# df = df.astype({"Full nucleus intensity mean": "float"})
# a = sns.jointplot(data=df, y="Full nucleus intensity mean", x="L1_compartment_difference", alpha = 0.1, hue= 'Cancer?')
# a = sns.jointplot(data=df, y="Line1_combined", x="L1_compartment_difference", alpha = 0.1, hue= 'Cancer?', kind = 'hex')
# plt.show()

# p = sns.jointplot(data=cdf, x = 'Line1_combined_cytoplasm', y='Line1_combined_nuclear', hue = "Cancer?", alpha = 0.3)
# plt.show()

#-----------------------------------------------------
#-------------------- Line1 by comparment, violins
ctypes = ["CAF", "cancer", "acinar","ductal","mDC","T.cell","macrophage"]
# cdf = df_chroma[df_chroma['Cell type'].isin(ctypes)].copy()
# cdf = cdf[["global_cid", "Cancer?","Cell type","Line1_combined_nuclear", "Line1_combined_cytoplasm","L1_compartment_ratio","L1_compartment_difference","L1_compartment_dif_ofTotal", "Line1_combined_nuclear_log10", "Line1_combined_cytoplasm_log10"]]
# cdf2 = pd.melt(cdf, id_vars = ["global_cid","Cell type"], value_vars  =["Line1_combined_nuclear","Line1_combined_cytoplasm","L1_compartment_difference","L1_compartment_dif_ofTotal"])
# cdf3 = pd.melt(cdf, id_vars = ["global_cid","Cancer?"], value_vars  =["Line1_combined_nuclear","Line1_combined_cytoplasm","L1_compartment_difference","L1_compartment_dif_ofTotal"])
# IPython.embed()
# p= sns.barplot(data=cdf2, x="Cell type", y="value", hue='variable')
# plt.show()
# cdf2 = pd.melt(cdf, id_vars = ["global_cid","Cell type"], value_vars  =["Line1_combined_nuclear","Line1_combined_cytoplasm"])
# # cdf3 = pd.melt(cdf, id_vars = ["global_cid","Cancer?"], value_vars  =["Line1_combined_nuclear","Line1_combined_cytoplasm","L1_compartment_difference","L1_compartment_dif_ofTotal"])
# p= sns.violinplot(data=cdf2, x="Cell type", y="value", hue='variable', split=True, cut = 0)
# # q= sns.violinplot(data=cdf3, x="Cancer?", y="value", hue='variable', split=True, cut = 0)
# plt.show()


#---------------------------------------------------
#--------------------- Chromatin ratio scatterplots 
# p = sns.scatterplot(data=df_chroma, x = 'Chromatin_intensity_ratio', y='Line1_combined', hue = "Cell type").set(title="Chromatin mean intensity ratio (outer/inner) vs LINE1")
# plt.show()


# p2 = sns.scatterplot(data=df_cancer, x = 'Chromatin_intensity_ratio', y='Line1_combined').set(title="Chromatin mean intensity ratio (outer/inner) vs LINE1, cancer only")
# plt.show()
# p = sns.scatterplot(data=df_chroma, x = 'Chromatin_intensity_ratio', y='Line1_combined', hue = "Cancer?", alpha = 0.3).set(title="Chromatin mean intensity ratio (outer/inner) vs LINE1")
# plt.show()

# p = sns.scatterplot(data=df_chroma_nd, x = 'Chromatin_intensity_ratio', y='Line1_combined', hue = "Cancer?", alpha = 0.3).set(title="Chromatin mean intensity ratio (outer/inner) vs LINE1")
# plt.show()
# IPython.embed()

#-----------------------------------------------------
##--------------- Barplots for chromatin ratio
# p = sns.barplot(data = df_chroma, x = 'Outer/inner_ratio_cut', y = 'Line1_combined', hue ='Cancer?')
# plt.show()
# IPython.embed()
# p = sns.histplot(data = df_chroma, x = 'Chromatin_intensity_ratio', hue = 'Cancer?',
#                  bins = histbins, multiple='stack')
# plt.show()

#-----------------------------------------------------
#---------------------- Pairplot
# g = sns.PairGrid(df_cancer2, vars=facet_vars, hue="Cancer?")
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)
# plt.show()

# jointplot
# p = sns.jointplot(data=df_chroma_nd, x = 'Chromatin_intensity_ratio', y='Line1_combined', hue = "Cancer?", alpha=0.3)
# plt.show()

df_comp = df_chroma.copy()
df_comp['compartment_ratio'] = df_comp["Line1_combined_nuclear"] / df_comp["Line1_combined_cytoplasm"]
df_comp = df_comp[df_comp["Nuclear area"] !="Too small"]
df_comp = df_comp.astype({'Nuclear area': 'float'})


# g = sns.FacetGrid(df_comp, col = "Cancer?")
# g.map_dataframe(sns.jointplot, x='Nuclear area',y = "Feret diameter min", hue = "Outer/inner_ratio_qcut", alpha = 0.3).set_titles(template="{col_name}")

#------------------- CAFs
# df_comp["CAF?"] = np.where(df_comp['Cell type'] == "CAF", "Yes","No")
# df_caf = df_comp[df_comp["CAF?"] == "Yes"].copy()
# df_caf["feret_ratio"] = df_caf["Feret diameter max"] / df_caf["Feret diameter min"]
# # p = sns.jointplot(data=df_caf, x='Eccentricity',hue = "Line1_combined_qcut",y = "Chromatin_intensity_ratio", alpha = 0.3)
# # plt.show()
# # IPython.embed()
# # df_cc = df_chroma.loc[df_chroma["Cell type"].isin(["CAF","cancer"])]
# df_allCAF = df.loc[df["Cell type"]=="CAF"].copy()
# df_allCAF['Line1_combined_qcut_CAFonly'] = pd.qcut(df_allCAF['Line1_combined'], 10)


#------------ KDEs
# cf = sns.jointplot(data=df_allCAF, x="Solidity", y="Eccentricity", kind="kde", joint_kws={'fill':True})
# cap = "Solidity vs Eccentricity in CAFS, KDE"
# cf.fig.suptitle(cap)


#----------------------------------------------------------
#---------------------------------------- Joinplots colored by L1
# a = sns.jointplot(data=df, x="Solidity", y="Eccentricity", 
#                    hue = "Line1_combined_qcut", alpha=0.1, palette = "rocket")
# cap = "Solidity vs Eccentricity in all cells, colored by LINE1 decile"
# a.fig.suptitle(cap)

# df_cancer = df.loc[df["Cancer?"]=="Cancer"].copy()
# df_cancer['Line1_combined_qcut_canceronly'] = pd.qcut(df_cancer['Line1_combined'], 10)

# c = sns.jointplot(data=df_cancer, x="Solidity", y="Eccentricity", 
#                    hue = "Line1_combined_qcut_canceronly", alpha=0.1, palette = "rocket")
# cap = "Solidity vs Eccentricity in Tumor, colored by LINE1 decile"
# c.fig.suptitle(cap)


# cf = sns.jointplot(data=df_allCAF, x="Solidity", y="Eccentricity", hue ="Line1_combined_qcut_CAFonly", alpha=0.1, palette = "rocket")
# cap = "Solidity vs Eccentricity in CAFS, colored by LINE1 decile"
# cf.fig.suptitle(cap)

# df_cc = df.loc[df["Cell type"].isin(["CAF","cancer"])]
# cc = sns.jointplot(data=df_cc, x="Solidity", y="Eccentricity", hue = "Cell type", alpha=0.1)
# cap = "Solidity vs Eccentricity in CAFS and Tumor only, colored by cell type"
# cc.fig.suptitle(cap)

# plt.show()

# p = sns.jointplot(data=df_caf, x='Chromatin_intensity_ratio', y = "Line1_combined", alpha = 0.3)


#-----------------------------------------------
#----------------------------- Faceted KDEs

ctypes = ["CAF", "cancer", "acinar","ductal","mDC","T.cell","macrophage", "Endothelial"]
cdf = df[df['Cell type'].isin(ctypes)].copy()
# cdf["L1_high_low"] = pd.qcut(cdf['Line1_combined'], 2, labels = ["L1 Low", "L1 High"])



# g = sns.FacetGrid(cdf, col = "Cell type", hue = "Cell type", col_wrap = 4)
# g.map(sns.kdeplot, "Solidity", "Eccentricity", fill  = True)

# ctypes = ["CAF", "cancer","T.cell","macrophage"]
# cdf = df[df['Cell type'].isin(ctypes)].copy()
# cdf["L1_high_low"] = pd.qcut(cdf['Line1_combined'], 2, labels = ["L1 Low", "L1 High"])
# g = sns.FacetGrid(cdf, col = "Cell type", hue = "Cell type", row = "L1_high_low")
# g.map(sns.kdeplot, "Solidity", "Eccentricity", fill  = True)

# #--------------- try a barplot with solidity cutoff
IPython.embed()
# cdf["Solidity_high_low"] = np.where(cdf['Solidity'] > 0.87, "Solidity High","Solidity Low")
# b = sns.barplot(data = cdf, x = 'Cell type', y = 'Line1_combined', hue ='Solidity_high_low')
# cap = "Line1 by cell type, colored by Solidity"
# b.fig.suptitle(cap)


## making something interesting for 5.25.23 Aryee lab meeting


runnames = ["B10_R1171_S2","C4_R5042_S1","D10_R5042_S2","Run5573_TMA1","Run5573_TMA28","Run5573_TMA31","Run5584_TMA32"]
basepath =r'/home/peter/home_projects/CosMx/All Runs/'
tx_extension = "_exprMat_file.csv"
transcripts = pd.DataFrame()
for i,n in enumerate(runnames):
    path = f'{basepath}{n}/{n}{tx_extension}'
    df = pd.read_csv(path)
    print(f"transcripts shape is {transcripts.shape}")
    df = df.astype({'fov': 'str', 'cell_ID':'str'})
    df.insert(0,"global_cid", "")
    df["global_cid"] = n + '_' + df.fov + '_' + df.cell_ID
    if i == 0:
        transcripts = df.copy()
    else:
        transcripts = pd.concat((transcripts,df), axis = 0)

cdf_with_transcripts = pd.merge(cdf, transcripts, how ='left', on = 'global_cid')
# print('running correlation...')
# transcript_cors = cdf_with_transcripts.corr(method = 'pearson', numeric_only=True)

# import scipy
# transcript_cors_pval = pd.DataFrame()
# corrs = []
# p_values = []
# sp_array = cdf_with_transcripts.select_dtypes(include=[np.number]).dropna()
# for feat in sp_array.columns:
#     corr, p_value = scipy.stats.pearsonr(sp_array['Solidity'], sp_array[feat])
#     corrs.append(corr)
#     p_values.append(p_value)

# transcript_cors_pval['variable'] = list(sp_array.columns)
# transcript_cors_pval['Correlation with Solidity'] = corrs
# transcript_cors_pval['p_value'] = p_values

# Differential expression by cell type


IPython.embed()
