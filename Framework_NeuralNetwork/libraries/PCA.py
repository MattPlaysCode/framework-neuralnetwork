import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pca(pca_names, plot_variances):
    ypos = np.arange(len(pca_names))
    plt.xticks(ypos, pca_names)
    plt.bar(pca_names, plot_variances)
    plt.xlabel("Cumulative Principal Components")
    plt.ylabel("Explained Variance in %")
    plt.title("Cumulated Vectors of PCA")
    plt.show()


class PCA:
    def __init__(self):
        self.index = 0
        self.res = pd.DataFrame()
        self.projected = []
        self.vectors = None

    def pca(self, X, sum_variance):
        cov_matrix = np.cov(X.T)
        values, self.vectors = np.linalg.eig(cov_matrix)
        explained_variances = []
        pca_names = []
        plot_variances = []
        for i in range(len(values)):
            explained_variances.append(values[i] / np.sum(values))  #values[i] / np.sum(values) relativiert die Varianzaufklärung eines Vektors an der Summe aller Vektoren (-> Werte zwischen 0 und 1)

        print(np.sum(explained_variances), '\n', explained_variances)

        if self.index == 0: # nur bei der ersten PCA die indizes festlegen
            sum = 0.0
            for i in range(len(explained_variances)):
                sum += explained_variances[i]
                if sum > sum_variance:
                    self.index = i
                    print("PCa vector count: " + str(self.index))
                    break
                if i == 0:
                    plot_variances.append(explained_variances[i])
                else:
                    plot_variances.append(explained_variances[i] + plot_variances[i - 1])
                pca_number = i + 1
                pca_names.append("PC" + str(pca_number))

        for i in range(self.index):
            self.projected.append(X.dot(self.vectors.T[i])) #hier werden die X_daten mit den Vektoren gedotproducted damit wir neue Spalten haben
            string = 'PC' + str(i)
            self.res[string] = self.projected[i]

        plot_pca(pca_names, plot_variances)

        return np.array(self.res)

    def pca_val(self, Y):
        out = []
        out_df = pd.DataFrame()
        for i in range(self.index):
            out.append(Y.dot(self.vectors.T[i]))
            string = 'PC' + str(i)
            out_df[string] = out[i]
        return out_df

