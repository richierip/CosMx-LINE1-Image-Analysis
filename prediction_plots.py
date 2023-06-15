import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import IPython



def plot_scatter(ref_path,pred_path):
    ref = pd.read_csv(ref_path)
    pred = pd.read_csv(pred_path)
    # IPython.embed()
    pred = pred.melt(id_vars=['fov']).rename(columns = {'variable':'feature','value':'predicted_Line1'})
    # IPython.embed()
    df = pd.merge(ref,pred, how = 'left',on = "fov")
    # IPython.embed()
    # p = sns.scatterplot(data=df, x="Line1_Combined", y="Predicted_Line1")
    # ax = sns.relplot(kind='scatter', x="Line1_Combined", y="Predicted_Line1", data=df, height=3.5, aspect=1.5)
    # ax.map_dataframe(sns.lineplot, 'x', 'y_line', color='g')
    plt.axline((0, 0), (1, 1), linewidth=4, color='#000000', alpha = 0.7)
    p = sns.lineplot(data=df, x="Line1_Combined", y="predicted_Line1", hue='feature')
    plt.show()
    g= sns.FacetGrid(data=df, hue='feature')
    g.map(plt.scatter,"Line1_Combined","predicted_Line1")
    g.map(sns.regplot,"Line1_Combined","predicted_Line1") 
    # IPython.embed()
    def const_line(*args,**kwargs):
        x = np.arange(0,115000,1000)
        y=x
        plt.plot(y,x,c='k')
    g.map(const_line)
    # for ax in g.axes_dict.values():
    #     ax.axline((0,0),slope=1,c=".2",ls="--",)
    # ab = plt.axline((0, 0), (1, 1), linewidth=4, color='#000000', alpha = 0.7)
    # g.map(ab)
    # g.add_legend()  
    plt.show()
    # pass

def main():
    ###### C4
    ref_path = r"/home/peter/home_projects/CosMx/C4_cancerOnly_Line1_sum_byFOV.csv"
    pred_path = r"/home/peter/home_projects/CosMx/C4_fov_predictions_100epochsD10_allPadded.csv"
    ###### D10
    # ref_path = r"/home/peter/home_projects/CosMx/D10_cancerOnly_Line1_sum_byFOV.csv"
    # pred_path = r"/home/peter/home_projects/CosMx/D10_fov_predictions_100epochsC4B10_allPadded.csv"
    ###### B10
    # ref_path = r"/home/peter/home_projects/CosMx/B10_cancerOnly_Line1_sum_byFOV.csv"
    # pred_path = r"/home/peter/home_projects/CosMx/B10_fov_predictions_250epochsC4D10_allPadded.csv"
    plot_scatter(ref_path,pred_path)


if __name__ == '__main__':
    main()