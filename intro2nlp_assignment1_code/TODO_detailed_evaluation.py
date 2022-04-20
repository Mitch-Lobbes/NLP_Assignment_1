# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
import json
import os
import subprocess

import train
import evaluate
import utils
import torch
import torch.optim as optim
import logging
from model.data_loader import DataLoader
import model.net as net


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


# !!! The file-paths need to be in the correct order --> "Random", "Majority", "Length", "Frequency", "LSTM" !!!

# lstm_output = ['experiments/base_model/model_output.tsv']
# ex12 = Detailed_Eval(lstm_output)


class Experiments:

    def __init__(self, epochs_values, filepath):
        self.f1_list = []
        self.filepath = filepath
        self.epochs_values = epochs_values
        self.epoch_experiment()

    def change_num_epochs(self, epoch: int):

        with open(self.filepath, 'r') as file:
            json_data = json.load(file)
            for item in enumerate(json_data):
                if item[1] == 'num_epochs':
                    json_data[item[1]] = epoch
        with open(self.filepath, 'w') as file:
            json.dump(json_data, file, indent=2)

    # def build_vocab(self):
    #    subprocess.run('python D:/build_vocab.py')

    def train_lstm(self):
        train.start()
        # subprocess.run('python D:/train.py')

    def evaluate_lstm(self, index):

        # Load the parameters
        args = evaluate.parser.parse_args()
        json_path = os.path.join(args.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = utils.Params(json_path)

        # use GPU if available
        params.cuda = torch.cuda.is_available()  # use GPU is available

        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)

        # Get the logger
        utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

        # Create the input data pipeline
        logging.info("Creating the dataset...")

        # load data
        data_loader = DataLoader(args.data_dir, params)
        data = data_loader.load_data(['test'], args.data_dir)
        test_data = data['test']

        # specify the test set size
        params.test_size = test_data['size']
        test_data_iterator = data_loader.data_iterator(test_data, params)

        logging.info("- done.")

        # Define the model
        model = net.Net(params).cuda() if params.cuda else net.Net(params)

        loss_fn = net.loss_fn
        metrics = net.metrics

        logging.info("Starting evaluation")

        # Reload weights from the saved file
        utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

        # Evaluate
        num_steps = (params.test_size + 1) // params.batch_size
        # MY ADJUSTMENTS
        # reverse the vocab and tag dictionary to be able to map back from ids to words and tags
        id2word = {v: k for k, v in data_loader.vocab.items()}
        tags = {v: k for k, v in data_loader.tag_map.items()}
        outfile = args.model_dir + f"/model_output{index}.tsv"
        test_metrics = evaluate.evaluate_and_output(model, loss_fn, test_data_iterator, metrics, num_steps, id2word,
                                                    tags,
                                                    outfile)
        save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
        utils.save_dict_to_json(test_metrics, save_path)

    def epoch_experiment(self):

        for i in range(len(self.epochs_values)):
            # changing the epoch parameter
            self.change_num_epochs(self.epochs_values[i])
            self.train_lstm()
            self.evaluate_lstm(i)

        for i in range(5):
            self.f1_calculation(f"experiments/base_model/model_output{i}.tsv")

        print(f'f1 results are: {self.f1_list}')

    def f1_calculation(self, filepath):

        file_output = pd.read_csv(filepath_or_buffer=filepath, encoding='latin-1',
                                  names=["Word", "Gold", "Prediction"],
                                  sep="\t")
        file_output = file_output.dropna(axis=0)

        weighted_avg_f1 = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                          average='weighted')
        self.f1_list.append(round(weighted_avg_f1[2],2))


epochs = [1, 3, 5, 7, 10]
json_path = "experiments/base_model/params.json"
x = Experiments(epochs, json_path)

# for number_of_epochs in range(len(self.epochs_values)):
#
#
#
# # Load the parameters from json file
# args = train_updated.parser.parse_args()
# json_path = os.path.join(args.model_dir, 'params.json')
# assert os.path.isfile(
#     json_path), "No json configuration file found at {}".format(json_path)
# params = utils.Params(json_path)
#
# # use GPU if available
# params.cuda = torch.cuda.is_available()
#
# # Set the random seed for reproducible experiments
# torch.manual_seed(230)
# if params.cuda:
#     torch.cuda.manual_seed(230)
#
# # Set the logger
# utils.set_logger(os.path.join(args.model_dir, 'train.log'))
#
# # Create the input data pipeline
# logging.info("Loading the datasets...")
#
# # load data
# data_loader = DataLoader(args.data_dir, params)
# data = data_loader.load_data(['train', 'val'], args.data_dir)
# train_data = data['train']
# val_data = data['val']
#
# # specify the train and val dataset sizes
# params.train_size = train_data['size']
# params.val_size = val_data['size']
#
# logging.info("- done.")
#
# # Define the model and optimizer
# model = net.Net(params).cuda() if params.cuda else net.Net(params)
# optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
#
# # fetch loss function and metrics
# loss_fn = net.loss_fn
# metrics = net.metrics
#
# # Train the model
# logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
# train_updated.train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir,
#                    args.restore_file)


epochs = [5, 10, 15, 20, 25]
