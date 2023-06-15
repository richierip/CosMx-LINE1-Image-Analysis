import pandas as pd
import IPython

def count_by_compartment(df):
    l1_cols = ["LINE1_ORF1","LINE1_ORF2"]
    categoricals = ["fov","cell_ID","target", "CellComp"]
    return df.loc[df["target"].isin(l1_cols),categoricals].groupby(by=categoricals).size().to_frame('target_count')


def main():
    print("Reading transcript CSV...\n")
    df = pd.read_csv(r"/home/peter/home_projects/CosMx/Other Runs/Run5584_TMA32/Run5584_TMA32_tx_file.csv")
    print("Grouping...\n")
    grouped_data = count_by_compartment(df)
    # IPython.embed()
    grouped_data.to_csv(r"/home/peter/home_projects/CosMx/Other Runs/Run5584_TMA32/LINE1_by_Compartment.csv")
    print("Done! New CSV saved in parent directory.\n")


if __name__ == "__main__":
    main()