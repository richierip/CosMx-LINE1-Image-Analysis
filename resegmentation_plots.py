import seaborn as sns
import numpy as np
import pandas as pd
from IPython import embed
import matplotlib.pyplot as plt


MIN_PANCK_THRESHOLD_OLD = 900
MIN_PANCK_THRESHOLD_NEW = 2100

old_metadata = pd.read_csv(r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\C4_R5042_S1_metadata_file.csv")
old_counts = pd.read_csv(r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\C4_R5042_S1_exprMat_file.csv")
reseg_metadata = pd.read_csv(r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\Resegmentation\C4_allFOV_cyto10_metadata.csv")
reseg_counts = pd.read_csv(r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\Resegmentation\C4_allFOV_cyto10_countsTable.csv")


reseg_counts = reseg_counts.loc[reseg_counts["cell_ID"] != 0] # get rid of cellID 0 rows (txs not assigned to any cell)
old_counts = old_counts.loc[old_counts["cell_ID"] != 0]

negprobes_columns = list(reseg_counts.filter(regex="NegPrb").columns)
krt_columns = list(reseg_counts.filter(regex="KRT").columns)
reseg_tx_per_cell = reseg_counts.drop(columns=["fov","cell_ID", *negprobes_columns]).sum().sum() / reseg_metadata.shape[0]
old_tx_per_cell = old_counts.drop(columns=["fov","cell_ID", *negprobes_columns]).sum().sum() / old_counts.shape[0]
old_counts["All NegProbes"] = old_counts[negprobes_columns].sum(axis=1)
reseg_counts["All NegProbes"] = reseg_counts[negprobes_columns].sum(axis=1)
old_counts["All KRT"] = old_counts[negprobes_columns].sum(axis=1)
reseg_counts["All KRT"] = reseg_counts[negprobes_columns].sum(axis=1)


# compare All KRT between images
old_metadata = old_metadata.merge(old_counts[["fov","cell_ID","All KRT","All NegProbes"]], how="left", on=["fov","cell_ID"])
old_metadata["Cancer?"] = np.where(old_metadata["Mean.PanCK"]>MIN_PANCK_THRESHOLD_OLD, "Cancer", "Other");
old_metadata["Dataset"] = "Old"
old_metadata = old_metadata.rename(columns={'Mean.PanCK':'Mean PanCK','Max.PanCK':'Max PanCK',
                                             'Mean.CD45':'Mean CD45', 'Max.CD45':'Max CD45',
                                            'Mean.CD3':'Mean CD3', 'Max.CD3':'Max CD3','Max.DAPI':'Max DAPI',
                            'Mean.DAPI':'Mean DAPI', "Area":"Cell area"}).drop(columns = ['Mean.MembraneStain', 'Max.MembraneStain'])

reseg_metadata = reseg_metadata.merge(reseg_counts[["fov","cell_ID","All KRT","All NegProbes"]], how="right", on=["fov","cell_ID"])
reseg_metadata["Cancer?"] = np.where(reseg_metadata["Mean PanCK"]>MIN_PANCK_THRESHOLD_NEW, "Cancer", "Other")
reseg_metadata["Dataset"] = "Resegmented"


merged = pd.concat([old_metadata[["fov","cell_ID", "Mean PanCK","Max PanCK", "Cancer?","Dataset", "All KRT","All NegProbes", "Cell area"]], 
                    reseg_metadata[["fov","cell_ID", "Mean PanCK","Max PanCK", "Cancer?","Dataset","All KRT","All NegProbes", "Cell area"]]]).reset_index(drop=True)

p = sns.barplot(data=merged, x = "Cancer?", y="All KRT", hue="Dataset")
p.set_title("All KRT genes mean transcripts per cell in Cancer vs. Other")
plt.show()
# embed()

# j = sns.jointplot(data=merged, x="Mean PanCK", y="All KRT", hue = "Dataset", kind = "kde", alpha = 0.8, fill=True)
# plt.show()
# fig, axs = plt.subplots(nrows=2)
# p = sns.histplot(data=old_metadata, x="All KRT", ax = axs[0], binwidth=1,binrange=(0,30) )
# p.set_title("Old data")
# q = sns.histplot(data=reseg_metadata, x="All KRT", ax = axs[1], binwidth = 1,binrange=(0,30) )
# q.set_title("Resegmented data")
# plt.subplot_tool()
# plt.show()


fig, axs = plt.subplots(nrows=2)
pa = sns.histplot(data=old_metadata, x="All KRT",hue="Cancer?" ,ax = axs[0], binwidth=1,binrange=(0,30), multiple= "dodge")
pa.set_title("Old data")
qa = sns.histplot(data=reseg_metadata, x="All KRT",hue="Cancer?", ax = axs[1], binwidth = 1,binrange=(0,30), multiple= "dodge" )
qa.set_title("Resegmented data")
plt.subplot_tool()
plt.show()


# Area normalized KRT
merged["KRT AreaNorm"] = merged["All KRT"] / merged["Cell area"]
p = sns.barplot(data=merged, x = "Cancer?", y="KRT AreaNorm", hue="Dataset")
p.set_title("Area normalized KRT genes mean transcripts per cell in Cancer vs. Other")
plt.show()

# Mean / Max PanCK

fig, axs = plt.subplots(ncols=2)
pa = sns.barplot(data=merged, x="Cancer?", y= "Mean PanCK", hue="Dataset",ax = axs[0])
pa.set_title("Mean PanCK old vs new")
qa = sns.barplot(data=merged, x="Cancer?", y= "Max PanCK", hue="Dataset",ax = axs[1])
qa.set_title("Max PanCK old vs new")
plt.subplot_tool()
plt.show()

p = sns.barplot(data=merged, x = "Cancer?", y="KRT AreaNorm", hue="Dataset")
p.set_title("Area normalized KRT genes mean transcripts per cell in Cancer vs. Other")
plt.show()

embed()