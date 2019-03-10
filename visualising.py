import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

LABELS = ['still', 'still', 'talking on the phone', 'writing on whiteboard',
          'drinking water', 'rinsing mouth with water',
          'brushing teeth', 'wearing contact lenses', 'wearing contact lenses',
          'talking on couch', 'relaxing on couch', 'cooking (chopping)', 'cooking (stirring)',
          'opening pill container', 'working on computer'
          ]
if __name__ == "__main__":
    matrix = np.random.randint(0, 100, (13, 13))

    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum

    # plot
    plt.switch_backend('agg')
    fig = plt.figure(figsize=[9.6, 7.2])
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    # fig.colorbar(cax)
    plt.imshow(matrix, interpolation='nearest',
               cmap=plt.cm.binary, aspect=0.7)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(i, j, str('%.1f' % matrix[j, i]),
                    va='center', ha='center', color='white')

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels(LABELS, rotation=45)
    ax.set_yticklabels(LABELS)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title('Confusion Matrix')
    plt.show()
    # save
    plt.savefig('fuck.jpg', dpi=200)
