import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator
import numpy as np

def draw_fig(score_list, save_fig_name):
    score_array = np.array(score_list)

    rank_rank_score_array = np.sort(score_array)

    processed_score_array = (1 - rank_rank_score_array) / 2 + 0.5

    f1 = np.array([rank_rank_score_array, processed_score_array])
    x1 = [list(range(0, rank_rank_score_array.shape[0]))]
    row_num = 1
    col_num = 1

    res = plt.figure(figsize=(19, 3.5))
    colorlist = ['b', 'r']
    labellist = ['processed_SiCF Score','Sample Weight: 1 - processed_SiCF Score']
    labellist2 = [r'$\gamma$: 0.1, 1, 10', r'$\tau: 0.05, 0.1, 0.15$', \
                  r'$\Phi_1: 0.05, 0.1, 0.15$', r'$\Phi_2: 0.25, 0.3, 0.35$']
    title_list = ['(1). processed SiCF scores and their weights for unlabeled and labeled samples',]

    x_label_lists = ['sample id']
    line_style_list = ['-', '-.']

    jsq = 0
    for row in range(row_num):
        # row = row + 1
        for col in range(col_num):
            # col = col + 1
            plt.subplot(row_num, col_num, jsq+1) 

            for i in range(len(labellist)):

                mid = f1[i]
                y = mid
                print(f"y={y}")
                y = np.transpose(y)

                plt.plot(x1[row*col_num+col], y, color=colorlist[i], linestyle=line_style_list[i], linewidth=1, marker='.', label=labellist[i])

            plt.legend(loc=4, fontsize=8)
            plt.ylabel('Sample Weight')
            plt.xlabel(x_label_lists[row*col_num+col])
            plt.grid(color="k", linestyle=":")
            plt.title(title_list[jsq])

            ax = plt.gca()
            ax.set_xticks([]) # remove x ids

            plt.legend(loc='lower left')

            jsq += 1


    plt.show()

    res.savefig(save_fig_name, bbox_inches='tight')
    print('save finished!')

