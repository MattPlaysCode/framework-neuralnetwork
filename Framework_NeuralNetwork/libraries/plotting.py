import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
import pandas as pd


def feature_error_comp(error_histories, architecture):
    fig, ax = plt.subplots(figsize=(14, 8))

    if isinstance(error_histories, dict):
        for key, value in error_histories.items():
            ax.plot(value, label=key)
    elif isinstance(error_histories, list):
        ax.plot(error_histories, label="Error_history")
    ax.legend()
    ax.set_xlabel("Iterationen")
    ax.set_ylabel("Error")
    ax.set_title("Features im Vergleich mit dem Array:________" + str(architecture))

    plt.show()


def bar_chart(f1_scores, x_axis_labels, architecture):
    if isinstance(f1_scores, dict):
        values_list = [[] for _ in range(len(x_axis_labels))]
        werte_dic = dict(zip(x_axis_labels, values_list))

        for k, v in f1_scores.items():  # HIER
            for i in range(len(v)):
                f1 = v[i]
                werte_dic[x_axis_labels[i]].append(f1)

        x = np.arange(len(x_axis_labels))  # the label locations

        fig, ax = plt.subplots(figsize=(15, 15))

        rects = []
        width = 0.10  # the width of the bars
        width_interval = width
        width_current = width_interval * (-1)

        v = list(f1_scores.values())
        labels_features = list(f1_scores.keys())
        for i in range(len(v)):  # in V[0] sind 6 f1 scores
            width_current += width_interval
            ax_bar = ax.bar(x + width_current, v[i], width)  # labels_features[i])
            rects.append(ax_bar)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('F1Scores')
        ax.set_title('F1-Score comparison in architecture: ____' + str(architecture) + ". Time : " + str(
            datetime.datetime.now().strftime("%X")))
        ax.set_xticks(np.arange(len(x_axis_labels)))
        ax.set_xticklabels(x_axis_labels)
        ax.legend(labels=labels_features)
        plt.show()
    elif isinstance(f1_scores, pd.DataFrame):
        f1_scores = f1_scores.iloc[:, 0].tolist()
        y_pos = np.arange(len(x_axis_labels))

        plt.bar(y_pos, f1_scores, align='center', alpha=0.5)
        plt.xticks(y_pos, x_axis_labels)
        plt.ylabel('F1Score')
        plt.title('F1 Scores of digits')

        plt.show()


def plot_confusion(confusion_matrizes, label_s, architecture):
    if confusion_matrizes.shape[0] < 1:
        for i in range(len(confusion_matrizes)):
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(confusion_matrizes[i], annot=True, ax=ax, fmt=".0f")
            ax.set_xlabel("prediction")
            ax.set_ylabel("ground truth")
            ax.set_title("Confusion_matrix for __" + label_s[i] + "__ in architecture " + str(architecture))
            plt.show()

    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(confusion_matrizes, annot=True, ax=ax, fmt=".0f")
        ax.set_xlabel("prediction")
        ax.set_ylabel("ground truth")
        ax.set_title("Confusion_matrix for __" + label_s + "__ in architecture " + str(architecture))
        plt.show()

