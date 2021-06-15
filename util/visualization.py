import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.manifold import TSNE
import seaborn as sns
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels


def gen_color_map(keys):
    colors = cm.nipy_spectral(np.linspace(0, 1, len(keys)))
    return dict(zip(keys, colors))

def visualize_dataset_2d(x1, x2, ys, alpha=0.5, x1_label='', x2_label='',
                         loc='upper left', figsize=(16, 8), xlim=None, ylim=None,
                         unique_ys=None, save_path=None, label_text_lookup=None):
    """
    Args:
    x1 - data's first dimention
    x2 - data's second dimention

    """

    # To avoid type 3 fonts. ACM Digital library complain about this
    # based on the recomendations here http://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if unique_ys is not None:
        c_lookup = gen_color_map(unique_ys)
    else:
        c_lookup = gen_color_map(set(ys))

    #     c_sequence = [''] * len(ys)
    #     for i in xrange(len(ys)):
    #         c_sequence[i] = c_lookup[ys[i]]

    plt.figure(figsize=figsize)
    for label in set(ys):
        color = c_lookup[label]
        mask = ys == label
        plt.scatter(x1[mask], x2[mask], c=color,
                    label=label if label_text_lookup is None else label_text_lookup[label],
                    alpha=alpha)

    #plt.scatter(x1, x2, c=c_sequence, alpha=alpha)
    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    lgd=plt.legend(loc=loc)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def qqplot(x, y, **kwargs):
    _, xr = stats.scatterplot(x, fit=False)
    _, yr = stats.scatterplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)

def unknown_dist_hist(X, ys, unknown_arr):
    df_all = pd.DataFrame({
     'z': X.tolist(),
     'label': ys
    })
   # df['z'] = df['z'].apply(lambda x: np.asarray(x))
  #  print(df.head(10))
    df = df_all[df_all.label.isin(unknown_arr)]
    overall_intra_dist = []
    overall_inter_dist = []
    for un in unknown_arr:
        df_tar = df[df['label']==un]
        df_non_tar = df[df['label']!=un]
        nd_df_tar = np.array([np.array(xi) for xi in df_tar['z'].values])
        nd_df_non_tar = np.array([np.array(xi) for xi in df_non_tar['z'].values])
   #     print(nd_df_tar)
    #    print(nd_df_non_tar)
        intra_dist_m = pairwise_distances(nd_df_tar)
        inter_dist_m = pairwise_distances(nd_df_tar, nd_df_non_tar)
        (r, c) = intra_dist_m.shape
        intra_dist = intra_dist_m[np.triu_indices(c, k = 1)]
        inter_dist = inter_dist_m.flatten()
      #  print(len(intra_ditst))
       # print(len(inter_dist))
        overall_intra_dist.extend(intra_dist)
        overall_inter_dist.extend(inter_dist)
        print("Max intra_distance: ")
        print(max(intra_dist))
        print("Min inter_distance: ")
        print(min(inter_dist))
        plt.figure()
        plt.title("Distribution of pairwise distances of digit " + str(un))
        sns.distplot(intra_dist, label="intra-distance")
        sns.distplot(inter_dist, label="inter-distance")
    #    plt.xlim(0, 40)
       # plt.ylim(0, 0.4)
        plt.legend()
    print("Max overall intra_distance: ")
    print(max(overall_intra_dist))
    print("Min overall inter_distance: ")
    print(min(overall_inter_dist))
    plt.figure()
    plt.title("Distribution of pairwise distances")
    sns.distplot(overall_intra_dist, label="intra-distance")
    sns.distplot(overall_inter_dist, label="inter-distance")
    #    plt.xlim(0, 40)
    plt.legend()
    plt.figure()
    plt.title("Distribution of pairwise distances (Zoom in)")
    sns.distplot(overall_intra_dist, label="intra-distance")
    sns.distplot(overall_inter_dist, label="inter-distance")
    #    plt.xlim(0, 40)
    plt.ylim(0, 0.2)
    plt.legend()

def visualize_t_SNE_for_train(X, ys, ts_known_mask=None, grid_shape=(2,2), alpha=0.5, xlim=None, ylim=None,
                    loc='upper left', bbox_to_anchor=(1.04,1), figsize=(5, 5),
                    unique_ys=None, save_path=None, label_text_lookup=None):

    plt.figure()
    print("t sne")
    print(len(X), len(ys))
    labels = []
    for label in ys:
        labels.append(label_text_lookup[label])

   # c_lookup = gen_color_map(label_text_lookup.values())
   # palette = sns.color_palette("bright", len(set(labels)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=80, random_state=1, n_iter=2000)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))


    tsne_df = pd.DataFrame({'t0': tsne_results[:, 0],
                            't1': tsne_results[:, 1],
                            'Category': labels})

    plt.figure()
    hue_order = sorted(set(labels))
    s = sns.FacetGrid(tsne_df, hue="Category", hue_order=hue_order, height=4)
    s.map(plt.scatter, "t0", "t1", alpha=.5, linewidth=.3, edgecolor="white")
    s.set(xlim=(-50, 50), ylim=(-50, 50))
    s.add_legend()
    print("here")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


def visualize_t_SNE(X, ys, ts_known_mask=None, grid_shape=(2,2), alpha=0.5, xlim=None, ylim=None,
                    loc='upper left', bbox_to_anchor=(1.04,1), figsize=(5, 5),
                    unique_ys=None, save_path=None, label_text_lookup=None):

    plt.figure()
    labels = []
    for label in ys:
        labels.append(label_text_lookup[label])

   # c_lookup = gen_color_map(label_text_lookup.values())
   # palette = sns.color_palette("bright", len(set(labels)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))


    tsne_df = pd.DataFrame({'t0': tsne_results[:, 0],
                            't1': tsne_results[:, 1],
                            'Category': labels,
                            'known': ts_known_mask})

    plt.figure()
    hue_order = sorted(set(labels))
    s = sns.FacetGrid(tsne_df, hue="Category", hue_order=hue_order, col="known", col_order=[True, False], height=4)
    s.map(plt.scatter, "t0", "t1", alpha=.5, linewidth=.3, edgecolor="white")
    s.set(xlim=(-50, 50), ylim=(-50, 50))
    s.add_legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


  #  s = sns.scatterplot(x="t0", y="t1", hue="digit", alpha=0.5,
      #        palette = c_lookup,
    #          legend='full',
      #        data=tsne_df)

   # s.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=1)
    #s.set(xlim=xlim,ylim=ylim)



# colors = c_lookup
 #   target_ids = range(len(ys))
   # for i, c, label in zip(target_ids, colors, set(ys)):
     #   plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=c, label=label if label_text_lookup is None else label_text_lookup[label])


  #  plt.legend()
  #  plt.show()

def visualize_dataset_nd(X, ys, grid_shape=(2,2), alpha=0.5, xlim=None, ylim=None,
                         loc='upper left', bbox_to_anchor=(1.04,1), figsize=(16, 8),
                         unique_ys=None, save_path=None, label_text_lookup=None):
    """
    Args:
    X: 2d np.array
    ys: 1d n.array

    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # To avoid type 3 fonts. ACM Digital library complain about this
    # based on the recomendations here http://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if unique_ys is not None:
        c_lookup = gen_color_map(unique_ys)
    else:
        c_lookup = gen_color_map(set(ys))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_shape[0], grid_shape[1])

    n_dim = X.shape[1]
    dim_1 = 0
    dim_2 = 1

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ax = fig.add_subplot(gs[i, j])
            for label in set(ys):
                color = c_lookup[label]
                mask = ys == label
                ax.scatter(X[mask, dim_1], X[mask, dim_2], c=color,
                           label=label if label_text_lookup is None else label_text_lookup[label],
                           alpha=alpha)
                ax.set_xlabel('Z{0}'.format(dim_1))
                ax.set_ylabel('Z{0}'.format(dim_2))
                ax.grid(True)
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)
            dim_2 += 1
            if dim_2 == n_dim:
                dim_1 += 1
                dim_2 = dim_1 + 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_z_separate(z, ts_y, ts_known_mask,
                      n_scatter=1000, unique_ys=range(7), xlim=None, ylim=None,
                      grid_shape=(1,3), figsize=(12, 4),font_size=13, markersize=None,
                      save_path_known=None,
                      save_path_unknown=None,
                      label_text_lookup=None):
    import matplotlib as mpl
    font = {'family' : 'normal',
    #         'weight' : 'bold',
            'size'   : font_size}

    mpl.rc('font', **font)

    def plot(z, ys, path):
        if z.shape[1] == 2:
            visualize_dataset_2d(z[:, 0], z[:, 1], ys,  xlim=xlim, ylim=ylim,
                                 alpha=0.5, figsize=(8, 6), unique_ys=unique_ys, save_path=path,
                                 label_text_lookup=label_text_lookup)
        elif z.shape[1] == 3:
            visualize_dataset_nd(z, ys, grid_shape=(1,3), alpha=0.5, xlim=xlim, ylim=ylim,
                                 loc='upper left', bbox_to_anchor=(1.04,1), figsize=(12, 4),
                                 unique_ys=unique_ys, save_path=path,
                                 label_text_lookup=label_text_lookup)
      #      visualize_t_SNE(z, ys, grid_shape=(1, 3), alpha=0.5, xlim=xlim, ylim=ylim,
         #                   loc='upper left', bbox_to_anchor=(1.04, 1), figsize=(12, 4),
        #                    unique_ys=unique_ys, save_path=path, label_text_lookup=None)


        else:
       #     visualize_dataset_nd(z, ys, grid_shape=grid_shape, alpha=0.5, xlim=xlim, ylim=ylim,
            #                     loc='upper left', bbox_to_anchor=(1.04,1), figsize=figsize,
           #                      unique_ys=unique_ys, save_path=path,
            #                     label_text_lookup=label_text_lookup)
            visualize_t_SNE(z, ys, grid_shape=grid_shape, alpha=0.5, xlim=xlim, ylim=ylim,
                            loc='upper left', bbox_to_anchor=(1.04, 1), figsize=figsize,
                            unique_ys=unique_ys, save_path=path, label_text_lookup=label_text_lookup)

    z = z[:n_scatter]
    y = np.argmax(ts_y[:n_scatter], axis=1)
    known_mask = ts_known_mask[:n_scatter]
    unknown_mask = np.logical_not(known_mask)
    #plot known
    plot(z[known_mask], y[known_mask], save_path_known)


    #plot unknown
    plot(z[unknown_mask], y[unknown_mask], save_path_unknown)



    mpl.rcdefaults()
