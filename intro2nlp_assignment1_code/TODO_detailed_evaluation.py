# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
import json
import os

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


#Save different F1 values for question 14

f = open('D:\\Users\\Paola\\Documents\\University\\Master\\NLP\Assignment_1\\NLP_Assignment_1\\intro2nlp_assignment1_code\\experiments\\base_model\\params.json')
params = json.load(f)
epochs = params["num_epochs"]
print(f"NN ran with {epochs} epochs.") 
f.close()

f_path = 'D:\\Users\\Paola\\Documents\\University\\Master\\NLP\Assignment_1\\NLP_Assignment_1\\intro2nlp_assignment1_code\\experiments\\base_model\\F1Values.json'

with open(f_path, "a+") as d:
    
    # check if size of file is 0
    if os.path.getsize(f_path) == 0:
        print('File is empty')

        fdict = {}
        fdict[epochs] = overall[2]
        json.dump(fdict, d)
    
    else:
        print('File is not empty')

        with open(f_path, "r+") as z:
            fdict = json.load(z)
            if str(epochs) not in fdict:
                fdict.update({epochs:overall[2]})
        with open(f_path, "w") as q:    
            json.dump(fdict,q)

