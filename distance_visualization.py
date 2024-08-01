import pandas as pd 
import numpy as np 
import json 
import matplotlib.pyplot as plt
from sklearn.manifold import MDS



#from stackover flow 
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns,
                          rowLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

if __name__ == '__main__':
    distances_json_path = '/mnt/storage/datasets/hila_cohen_DTI/outputs/distances_json/240721/124614_50_points_1_lambda.json'
    distances_json_wolf = '/mnt/storage/datasets/hila_cohen_DTI/outputs/distances_json/240724/134009_50_points_1_lambda.json'
    distances_dict = json.load(open(distances_json_path))
    distances_dict_wolf = json.load(open(distances_json_wolf))
    distances_dict.update(distances_dict_wolf)
    Animals_to_check = ['Dog',
                        'Hyrax',
                        'WildRat2',
                        'Cow1',
                        'Giraffe1',
                        'Orangutan1',
                        'Donkey',
                        'Chimpanzee',
                        'Horse1',
                        'Wolf1']
    df_distances = pd.DataFrame(columns=Animals_to_check, index=pd.Index(Animals_to_check))
    df_distances = df_distances.fillna(0)

    for key in distances_dict.keys():
        current_distance = round(distances_dict[key], 2)
        animal_1 = key.split('_')[0]
        animal_2 = key.split('_')[2]
        df_distances.loc[animal_1, animal_2] = current_distance
        df_distances.loc[animal_2, animal_1] = current_distance
    fig,ax = render_mpl_table(df_distances, header_columns=0, col_width=2.0)

    fig.savefig("table_distance.png")

    # Create an MDS model with the desired number of dimensions
    # Number of dimensions for visualization
    n_components = 2 
    mds = MDS(n_components=n_components, normalized_stress=False, dissimilarity='precomputed')
    # Fit the MDS model to your data
    X_reduced = mds.fit_transform(df_distances)
    # Visualize the dms data
    fig, ax = plt.subplots()
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1],)
    for i, txt in enumerate(Animals_to_check):
        ax.annotate(txt, (X_reduced[i, 0], X_reduced[i, 1]))
    plt.title("MDS Visualization")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.savefig("mds.png")


