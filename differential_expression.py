'''
Differential expression on CosMx runs (3 slides, 4 TMAS)
Slide names: ["B10_R1171_S2","C4_R5042_S1","D10_R5042_S2","Run5573_TMA1","Run5573_TMA28","Run5573_TMA31","Run5584_TMA32"]

Looking to see a difference between expression in cells binned by a Solidity threshold

Useful scanpy example: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
'''


import diffxpy.api as de
import numpy as np
import pandas as pd
import seaborn as sns
import IPython
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
sc.settings.verbosity = 2 # 0 = errors, 1 = warnings, 2 = info, 3 = hints
from adjustText import adjust_text

'''-------------- Read data, add columns '''
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
df['Line1_combined_qcut']= pd.qcut(df['Line1_combined'], 10)

# make one with selected cell types only
ctypes = ["CAF", "cancer", "acinar","ductal","mDC","T.cell","macrophage", "Endothelial"]
cdf = df[df['Cell type'].isin(ctypes)].copy()
cdf["Solidity_high_low"] = np.where(cdf['Solidity'] > 0.89, "Solidity High","Solidity Low")

'''-------------- Get transcripts and merge'''
# runnames = ["B10_R1171_S2","C4_R5042_S1","D10_R5042_S2","Run5573_TMA1","Run5573_TMA28","Run5573_TMA31","Run5584_TMA32"]
# basepath =r'/home/peter/home_projects/CosMx/All Runs/'
# tx_extension = "_exprMat_file.csv"
# transcripts = pd.DataFrame()
# for i,n in enumerate(runnames):
#     path = f'{basepath}{n}/{n}{tx_extension}'
#     df = pd.read_csv(path)
#     print(f"transcripts shape is {transcripts.shape}")
#     df = df.astype({'fov': 'str', 'cell_ID':'str'})
#     df.insert(0,"global_cid", "")
#     df["global_cid"] = n + '_' + df.fov + '_' + df.cell_ID
#     if i == 0:
#         transcripts = df.copy()
#     else:
#         transcripts = pd.concat((transcripts,df), axis = 0)

# cdf_with_transcripts = pd.merge(cdf, transcripts, how ='left', on = 'global_cid')

# cdf_transcripts_only = cdf_with_transcripts.iloc[:,64:-20]

# # When the pipeline is figured out, do some normalization... 
# #TODO


# # Doing this makes AnnData read in the global ID as the observation labels
# cdf_transcripts_only.insert(0,"global_cid", cdf_with_transcripts["global_cid"])
# cdf_transcripts_only.to_csv("/home/peter/home_projects/CosMx/Differential Expression/cdf_expressionMatrix.csv")
adata = ad.read_csv("/home/peter/home_projects/CosMx/Differential Expression/cdf_expressionMatrix.csv") # Reading takes a minute or two.

'''-------------- Add metadata as observations'''
for metric in ["Solidity_high_low", "Cell type"]:
    adata.obs[metric] = pd.Categorical(cdf[metric])
adata.obs['Total_transcript_counts'] = np.array(cdf['Total transcript counts'].astype('int'))


'''-------------- Pre function calls '''

sc.pp.calculate_qc_metrics(adata, percent_top=None,log1p=False, inplace=True)

#sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4)
#sc.pl.scatter(adata, x = 'total_counts', y='n_genes_by_counts', size = 10, alpha = 0.3, color = 'Cell type')

#Slice based on these numbers
sc.pp.filter_genes(adata, min_cells = 1000) # No genes get caught by this 
sc.pp.filter_cells(adata, min_genes = 15) # removes 500 cells
adata = adata[adata.obs.n_genes_by_counts < 500,:].copy() # #TODO how to get this done above?

'''-------------- Add normalized layers / normalize X'''
# cdf['Total transcript counts'].astype('int').mean()       >> 221.97

# adata.layers["log_transformed"] = np.log1p(adata.X)
# adata.layers["norm_counts"] = sc.pp.normalize_total(adata, target_sum=1e3,inplace=False)['X']
# adata.layers["log_norm_counts"] = np.log1p(adata.layers["norm_counts"])

adata.layers["original"] = adata.X
sc.pp.normalize_total(adata, target_sum=1e3)
sc.pp.log1p(adata)


# ------- Create new subsets 
#TODO by celltype!!
# for ct in ctypes:
cdata = adata[adata.obs['Cell type'] == 'cancer'].copy()
cdata_low = cdata[cdata.obs['Solidity_high_low'] == 'Solidity Low'].copy()
cdata_high = cdata[cdata.obs['Solidity_high_low'] == 'Solidity High'].copy()
#----------- rank gene variability
sc.pp.highly_variable_genes(adata) #, flavor = 'seurat_v3', n_top_genes=965, layer='original') # This part bonks...

#----------- more QC plots for variability
# ax = sc.pl.highest_expr_genes(adata, show = False)
# ax.set_title("Highest expressing genes in CosMx subset - all runs, Cancer/CAF/acinar/ductal/mDC/Tcell/macrophage/endothelial")
# ax2 = sc.pl.highly_variable_genes(adata, show = False)

#------------ Filter to highly variable only
# adata = adata[:,adata.var.highly_variable].copy()
# sc.pp.regress_out(adata, ['total_counts'])
# sc.pp.scale(adata, max_value=10)

#----------- PCA whole dataset
# sc.tl.pca(adata)
# sc.pl.pca(adata, color = "SERPINA1")
# sc.pl.pca_variance_ratio(adata, log=True)


adata.raw = adata


'''#!!!!!!!!!!!!!!!! Switching to cancer only here.'''
cdata.raw = cdata
#------------ Filter to highly variable only
sc.pp.highly_variable_genes(cdata)
cdata = cdata[:,cdata.var.highly_variable].copy()
# sc.pp.regress_out(adata, ['total_counts'])
# sc.pp.scale(cdata, max_value=10)

# PCAs again on just cancer

sc.tl.pca(cdata)
# sc.pl.pca(cdata, color = "SERPINA1")
# sc.pl.pca_variance_ratio(cdata, log=True)
sc.pp.neighbors(cdata, n_neighbors=10, n_pcs=40)
sc.tl.umap(cdata)
# sc.pl.umap(cdata, color='Solidity_high_low', size=10)
sc.tl.leiden(cdata)

print("\nBefore ranking...\n")
IPython.embed()
# Differential expression
sc.tl.rank_genes_groups(cdata, 'Solidity_high_low', method='wilcoxon')
# sc.tl.rank_genes_groups(cdata, 'Solidity_high_low', layer = 'log_norm_counts', method='wilcoxon',tie_correct=True, corr_method="benjamini-hochberg")



# Make frame for plotting
df = sc.get.rank_genes_groups_df(cdata, group = "Solidity High")
    # df = df[df["scores"] > 0] 


# heatmaps / matrix plots
# top30_variable = adata.var.sort_values(by='highly_variable_rank').index[:30]
# ctop30_variable = cdata.var.sort_values(by='highly_variable_rank').index[:30]
# sc.pl.matrixplot(cdata, top30_variable, groupby = "Solidity_high_low",layer = "log_transformed", swap_axes = True)


# Fix underflow zeros in pvals
df.loc[df['pvals_adj'] == 0,["pvals","pvals_adj"]] = 1e-310
cdata_low = cdata[cdata.obs['Solidity_high_low'] == 'Solidity Low'].copy()
cdata_high = cdata[cdata.obs['Solidity_high_low'] == 'Solidity High'].copy()

print("\nStopping before volcano")
IPython.embed()

'''#------------- MA plot: mean expression vs LFC'''
# cdata = adata[adata.obs['Cell type'] == 'cancer'].copy()
cdata_low = cdata[cdata.obs['Solidity_high_low'] == 'Solidity Low'].copy()
cdata_high = cdata[cdata.obs['Solidity_high_low'] == 'Solidity High'].copy()
mean_expr_low = np.mean(cdata_low.layers['original'], axis=0)
mean_expr_high = np.mean(cdata_high.layers['original'], axis=0)
ms_df = pd.DataFrame(mean_expr_low, columns = ['low_solidity_expression'])
ms_df["high_solidity_expression"] = mean_expr_high
ms_df['names'] = cdata.raw.var_names
ms_df['log_expr_low'] = np.log1p(ms_df['low_solidity_expression'])
ms_df['log_expr_high'] = np.log1p(ms_df['high_solidity_expression'])

# lfc = cdata.uns['rank_genes_groups']['logfoldchanges']['Solidity High']
ms_df = pd.merge(ms_df, df, how = 'left', on = 'names')

#plotting
ax = plt.axes()
p = sns.scatterplot(data = ms_df,x = 'log_expr_low', y = 'logfoldchanges', color='blue', ax=ax, alpha = 0.5, size = 2)
p = sns.scatterplot(data = ms_df,x = 'log_expr_high', y = 'logfoldchanges', color='red', ax=ax, alpha = 0.5, size=2)
p.set_title("MA plot of high solidity (red) vs low solidity (blue), superimposed")
plt.axhline(0,color="black",linestyle="--")
plt.ylabel("Log2FC (high compared to low)")
plt.xlabel("Log1P Mean Expression")
plt.show()


#--------- plotting the volcano

plt.figure(figsize=(15,10))
plt.scatter(x=df['logfoldchanges'],y=df['pvals_adj'].apply(lambda x:-np.log10(x)),s=5,label="Not significant")

# highlight down- or up- regulated genes
hthresh = list(df['logfoldchanges'].sort_values())[-10]
lthresh = list(df['logfoldchanges'].sort_values())[10]
down = df[(df['logfoldchanges']<=lthresh)&(df['pvals_adj']<=0.001)]
up = df[(df['logfoldchanges']>=hthresh)&(df['pvals_adj']<=0.001)]
special = df[df['names'].isin(["LINE1_ORF1", "LINE1_ORF2"])]

plt.scatter(x=down['logfoldchanges'],y=down['pvals_adj'].apply(lambda x:-np.log10(x)),s=10,label="Down-regulated in circular cells (CANCER)",color="blue")
plt.scatter(x=up['logfoldchanges'],y=up['pvals_adj'].apply(lambda x:-np.log10(x)),s=10,label="Up-regulated in circular cells (CANCER)",color="red")
plt.scatter(x=special['logfoldchanges'],y=special['pvals_adj'].apply(lambda x:-np.log10(x)),s=40,label="LINE1",color="green")
uptexts = []
downtexts = []
for i,r in up.iterrows():
    uptexts.append(plt.text(x=r['logfoldchanges'],y=-np.log10(r['pvals_adj']),s=r['names']))
for i,r in down.iterrows():
    downtexts.append(plt.text(x=r['logfoldchanges'],y=-np.log10(r['pvals_adj']),s=r['names']))
adjust_text(uptexts+downtexts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

plt.xlabel("Log2FC")
plt.ylabel("-log10 p-adj")
plt.axvline(lthresh,color="grey",linestyle="--")
plt.axvline(hthresh,color="grey",linestyle="--")
plt.axhline(2,color="grey",linestyle="--")
plt.legend()
plt.show()


print("\nDone with everything")
IPython.embed()