# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
import json
import os


class Detailed_Eval:

    def __init__(self, filelist):
        self.filelist = filelist
        self.model_list = ["Random", "Majority", "Length", "Frequency", "LSTM"]
        self.table = pd.DataFrame(columns=["Model",
                                           "Class_N Precision",
                                           "Class_N Recall",
                                           "Clgit ass_N F1",
                                           "Class_C Precision",
                                           "Class_C Recall",
                                           "Class_C F1",
                                           "Weighted Average F1"])
        self.update_table()
        self.print_table()

    def update_table(self):
        for filepath in enumerate(self.filelist):
            file_output = pd.read_csv(filepath_or_buffer=filepath[1], encoding='latin-1',
                                      names=["Word", "Gold", "Prediction"],
                                      sep="\t")
            file_output = file_output.dropna(axis=0)

            C_metrics = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                        average='binary', pos_label='C')
            N_metrics = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                        average='binary', pos_label='N')
            weighted_avg_f1 = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                              average='weighted')
            self.table.loc[len(self.table.index)] = [self.model_list[filepath[0]], round(N_metrics[0], 2),
                                                     round(N_metrics[1], 2), round(N_metrics[2], 2),
                                                     round(C_metrics[0], 2), round(C_metrics[1], 2),
                                                     round(C_metrics[2], 2), round(weighted_avg_f1[0], 2)]

    def print_table(self):
        print("Detailed Evaluation (2.5 points)")
        print(self.table.to_string())
        print("\n")


# !!! The files need to be in the correct order --> "Random", "Majority", "Length", "Frequency", "LSTM" !!!

lstm_output = ['experiments/base_model/model_output.tsv']
ex12 = Detailed_Eval(lstm_output)

