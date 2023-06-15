import diffxpy.api as de
import numpy as np
import pandas as pd
import seaborn as sns
import IPython
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from adjustText import adjust_text

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

# # Get transcripts and merge
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

# add metadata as observations
for metric in ["Solidity_high_low", "Cell type"]:
    adata.obs[metric] = pd.Categorical(cdf[metric])
adata.obs['Total_transcript_counts'] = np.array(cdf['Total transcript counts'].astype('int'))


'''Add normalized layers - should look something like this'''
# cdf['Total transcript counts'].astype('int').mean()       >> 221.97
# adata.layers["log_normed_by_nTranscripts"] = np.log1p(adata.obs["Total_transcript_counts"])

adata.layers["log_transformed"] = np.log1p(adata.X)
adata.layers["norm_counts"] = sc.pp.normalize_total(adata, target_sum=1e3,inplace=False)['X']
adata.layers["log_norm_counts"] = np.log1p(adata.layers["norm_counts"])

#--------------------- QC function calls
sc.pp.calculate_qc_metrics(adata)


# rank gene variability
sc.pp.highly_variable_genes(adata, flavor = 'seurat_v3', n_top_genes=965)

#----------- QC plots
ax = sc.pl.highest_expr_genes(adata, show = False)
ax.set_title("Highest expressing genes in CosMx subset - all runs, Cancer/CAF/acinar/ductal/mDC/Tcell/macrophage/endothelial")
ax2 = sc.pl.highly_variable_genes(adata, show = False)

print("Embedding before making cancer only anndata object")
IPython.embed()
# ------- Save differential expression results
#TODO by celltype!!
# for ct in ctypes:
cdata = adata[adata.obs['Cell type'] == 'cancer'].copy()
cdata_low = cdata[cdata.obs['Solidity_high_low'] == 'Solidity Low'].copy()
cdata_high = cdata[cdata.obs['Solidity_high_low'] == 'Solidity High'].copy()
sc.tl.rank_genes_groups(cdata, 'Solidity_high_low', layer = 'log_norm_counts', method='wilcoxon',tie_correct=True, corr_method="benjamini-hochberg")

# Make frame for plotting
df = sc.get.rank_genes_groups_df(cdata, group = "Solidity High")
    # df = df[df["scores"] > 0] 


# heatmaps / matrix plots
top30_variable = adata.var.sort_values(by='highly_variable_rank').index[:30]
ctop30_variable = cdata.var.sort_values(by='highly_variable_rank').index[:30]
sc.pl.matrixplot(cdata, top30_variable, groupby = "Solidity_high_low",layer = "log_transformed", swap_axes = True)


# Fix underflow zeros in pvals
df.loc[df['pvals_adj'] == 0,["pvals","pvals_adj"]] = 1e-310


print("Stopping before volcano")
IPython.embed()

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

IPython.embed()