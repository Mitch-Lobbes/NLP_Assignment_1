# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import f1_score

results = pd.read_csv(filepath_or_buffer='D:\\Users\\Paola\\Documents\\University\\Master\\NLP\\Assignment_1\\NLP_Assignment_1\\intro2nlp_assignment1_code\\experiments\\base_model\\model_output.tsv', encoding='latin-1', names=["Word", "Gold", "Prediction"], sep="\t")

print(results) #data contains some useless rows

results = results.dropna(axis=0)
print(results)

C_metrics = precision_recall_fscore_support(results["Gold"], results["Prediction"], average='binary', pos_label='C')
N_metrics = precision_recall_fscore_support(results["Gold"], results["Prediction"], average='binary', pos_label='N')
overall = precision_recall_fscore_support(results["Gold"], results["Prediction"], average='weighted')
print("\nEvaluation metrics for class C:\n---------------------------------------------------------------------\n", C_metrics)
print("\nEvaluation metrics for class N:\n---------------------------------------------------------------------\n", N_metrics)
print("\nWeighted average F1 score:", overall[2])