import numpy as np
import pandas as pd


class Calculus:
    def sum(self):
        df = pd.read_csv(
            r'C:\Users\DELL\Desktop\8 th semester\CO 544 - Machine Learning and Data mining\Labs\Lab 01\Data Sets\labExercise01.csv',
            header=None)

        std = []
        for c in range(0, 5):
            column = df.iloc[0:, c]
            mean = np.mean(column)
            total = 0

            for i in range(0, column.size):
                difference = column[i] - mean
                total = total + np.power(difference, 2)

            total = total / (column.size - 1)
            total = np.sqrt(total)

            std.append(total)

        print("Standard deviations of each column of the matrix are : ")
        return std
