import torch
import torch.nn as nn
import numpy as np
import math

from pathlib import Path

from webcam_eyetracking.geddnet import GEDDnet


import matplotlib.pyplot as plt
import seaborn as sns

import logging
import sys

PROJECT_NAME = "webcam_eyetracking_dl"


class ModelTrainingHelper():

    def __init__(self, hp_config):
        self.hp_config = hp_config
    
    def get_training_device(self, gpu_num=0):
        # Device selection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if gpu_num > torch.cuda.device_count():
            print(f"Supplied GPU {gpu_num} is greater than available count {torch.device_count()}")
            gpu_num = 0

        if device == 'cuda':
            device = "cuda:" + str(gpu_num)
        
        return device
    
    def init_model(self, from_path=None, vgg16_path=None, random_seed=42):

        if self.hp_config["model_type"] == "geddnet":
            model = GEDDnet(vgg16_path=vgg16_path,
                            dropout_p=self.hp_config["dropout"], 
                            init_weights=self.hp_config["init_geddnet"], 
                            use_pbias=self.hp_config["use_pbias"])

        if from_path is None:
            return model
        else:
            self.load_model_from_file(model, from_path)
            return model

    # Function to save the trained model in a checkpoint file.
    def save_model(self, model, model_save_path):
        torch.save(model.state_dict(), model_save_path)

    def load_model_from_file(self, model, model_path):
        model.load_state_dict(torch.load(model_path))

    
    #Method for computing the number of predictions greater than 5% of diagonal distance. 
    def diagonal_dist_metric(self, y, y_hat, height_tensor, width_tensor):
        
        # Find difference in X and Y co-ordinates.
        # Note that we multiply scale up by multiplying with width and height
        w_diff = (y[:, 0] - y_hat[:, 0])*width_tensor
        h_diff = (y[:, 1] - y_hat[:, 1])*height_tensor

        # Square of the difference
        w_diff = (w_diff.pow(2))
        h_diff = (h_diff.pow(2))

        # Compute the square root and sum up the values over all frames, participants 
        total_euc_dist = (w_diff + h_diff).sqrt()

        # Diagonal Distances
        diag_dtances = (width_tensor.pow(2) + height_tensor.pow(2)).sqrt()

        return (total_euc_dist > self.hp_config["diagonal_dist_ratio"]*diag_dtances).sum().item()


    #Method for computing the EUclidean distance berween the predicted gaze and actual gaze
    def evaluation_metric(self, y, y_hat, height_tensor, width_tensor):
        # Euclidean distance between the predicted and the actual gaze coordinates

        # Find difference in X and Y co-ordinates.
        # Note that we multiply scale up by multiplying with width and height
        w_diff = (y[:, 0] - y_hat[:, 0])*width_tensor
        h_diff = (y[:, 1] - y_hat[:, 1])*height_tensor

        # Square of the difference
        w_diff = (w_diff.pow(2))
        h_diff = (h_diff.pow(2))

        # Compute the square root and sum up the values over all frames, participants 
        total_euc_dist = (w_diff + h_diff).sqrt().sum().item()

        return total_euc_dist
    
    def  plot_train_test_metric_vs_epochs(self, train_metric, test_metric, figure_save_path,
                                          metric, title=None, save = True):

        epochs = [epoch for epoch in range(self.hp_config["plot_epoch_start"]+1, 
                    self.hp_config["num_epochs"])]

        plt.close('all')
        fig, ax = plt.subplots(tight_layout=True)

        p1,  = ax.plot(epochs, train_metric,  color='blue', marker='o', label='Train ' + metric)
        p2,  = ax.plot(epochs, test_metric,  color='red', marker='o', label='Test ' + metric)

        plt_title = title or 'Loss Plot'
        ax.set_xlabel("Epoch Count")
        ax.set_ylabel(metric)
        ax.set_title(plt_title)

        ax.legend(handles=[p1, p2], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

        plt.show()
        
        if save:
            fig.savefig(figure_save_path)
        plt.close('all')
    
    def plot_heatmap(self, bin_np, columns, metric, figure_save_path):
        plt.close('all')
        plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
        ax = sns.heatmap(bin_np, linewidth=0.5, xticklabels=columns, yticklabels=columns, annot=True, fmt='g', annot_kws={'fontsize':8}).set(title="Heat Map " + metric)
        plt.savefig(figure_save_path)
        plt.show()


    def populate_bins(self, val_np, bin_np):
        for x,y in val_np:
            # Note that we switch up x,y here since Tobii co-ordinates "x, y" are "y, x" in 
            # seaborns plotting package.

            bin_np[math.floor(y*10) if y < 1.0 else -1][math.floor(x*10) if x < 1.0 else -1] += 1 


    def set_hp(self, hp_config_new):
        self.hp_config = hp_config_new
    

    

class ModelPathUtils():
    def __init__(self):

        self.model_results_path = self.get_project_root() / "reports"
        self.exp_pth_dict = {}

    def create_or_update_model_train_paths(self, experiment_name="EOTT_Exp", 
                                            config_num=0, run_num=0):
        exp_op_path = self.model_results_path / experiment_name
        exp_op_path.mkdir(exist_ok=True, parents=True)

        config_op_path = exp_op_path / f"config{config_num}"
        config_op_path.mkdir(exist_ok=True, parents=True)

        run_op_path = config_op_path / f"runs{run_num}"
        run_op_path.mkdir(exist_ok=True, parents=True)

        figures_op_path = run_op_path / "Figures"
        figures_op_path.mkdir(exist_ok=True, parents=True)

        checkpoint_op_path = run_op_path / "ckpt"
        checkpoint_op_path.mkdir(exist_ok=True, parents=True)

        loss_file_path = run_op_path / "loss_file.txt"
        metric_file_path = run_op_path / "euc_metric_file.txt"
        diag_metric_file_path = run_op_path / "diag_metric_file.txt"

        self.exp_pth_dict["experiment_name"] = experiment_name
        self.exp_pth_dict["run_count"] = run_num
        self.exp_pth_dict["config_num"] = config_num
        self.exp_pth_dict["exp_op_path"] =  exp_op_path
        self.exp_pth_dict["config_op_path"] =  config_op_path
        self.exp_pth_dict["run_op_path"] =  run_op_path
        self.exp_pth_dict["figures_op_path"] =  figures_op_path
        self.exp_pth_dict["checkpoint_op_path"] =  checkpoint_op_path
        self.exp_pth_dict["loss_file_path"] =  loss_file_path
        self.exp_pth_dict["metric_file_path"] =  metric_file_path
        self.exp_pth_dict["diag_metric_file_path"] = diag_metric_file_path

        return self.exp_pth_dict
    
    def get_processed_dataset_path(self):
        return self.get_project_root() / "data" / "processed"

    def get_project_root(self):
        project_root_dir = [p for p in Path().resolve().parents 
                        if p.parts[-1].lower()== PROJECT_NAME.lower()][0]
        return project_root_dir


class  LoggingHelper():

    def __init__(self):
        self.formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                            '%m-%d-%Y %H:%M:%S')

    def get_logger(self, name, log_level=None):
        logger = logging.getLogger(name)
        log_level = log_level if log_level else logging.DEBUG
        logger.setLevel(logging.INFO)
        return logger

    def add_handler(self, logger, stdout=True, file_path=None):
        if stdout:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(self.formatter)
            logger.addHandler(stdout_handler)

        if file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)
    
    def clear_handlers(self, logger):
        logger.handlers = []
    
