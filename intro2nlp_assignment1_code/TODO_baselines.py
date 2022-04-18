# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader
import torch

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

wiki_news_train = pd.read_csv("D:\\Users\\Paola\\Documents\\University\\Master\\NLP\\Assignment_1\\NLP_Assignment_1\\intro2nlp_assignment1_code\\data\\original\\english\\WikiNews_Train.tsv", sep="\t", header=None, encoding='utf-8')


# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    # TODO: determine the majority class based on the training data
    testinput.split('').count("N")

    majority_class = "N"
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append((instance, instance_predictions))

    # TODO: calculate accuracy for the test input
    # ...
    return None, predictions


if __name__ == '__main__':
    train_path = "D:\\Users\\Paola\\Documents\\University\\Master\\NLP\\Assignment_1\\NLP_Assignment_1\\intro2nlp_assignment1_code\\data\\preprocessed\\train\\"
    dev_path =  "D:\\Users\\Paola\\Documents\\University\\Master\\NLP\\Assignment_1\\NLP_Assignment_1\\intro2nlp_assignment1_code\\data\\preprocessed\\val\\"
    test_path =  "D:\\Users\\Paola\\Documents\\University\\Master\\NLP\\Assignment_1\\NLP_Assignment_1\\intro2nlp_assignment1_code\\data\\preprocessed\\test\\"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
    with open(train_path + "sentences.txt", encoding='utf-8') as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt", encoding='utf-8') as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt", encoding='utf-8') as dev_file:
        dev_sentences = dev_file.readlines()

    with open(train_path + "labels.txt", encoding='utf-8') as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt", encoding='utf-8') as testfile:
        testinput = testfile.readlines()

    with open(test_path + "labels.txt", encoding='utf-8') as test_label_file:
        testlabels = test_label_file.readlines()


    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, testinput, testlabels)

    # TODO: output the predictions in a suitable way so that you can evaluate them