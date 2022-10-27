# %%
import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import json

from pathlib import Path

from sklearn.model_selection import KFold

from torch.utils.data import DataLoader
import torch.optim as optim

from webcam_eyetracking._utils import get_project_root
from webcam_eyetracking.models.utils import LoggingHelper, ModelTrainingHelper, ModelPathUtils
from webcam_eyetracking.models.gaze_dataset import GazeDataset
from webcam_eyetracking.models.geddnet import GEDDnet


# %%
def execute_step(dataset, phase,  epoch,  fold_num,  log_loss=False, log_evaluation_metric=False, sampler=None):
    

    dataloader = DataLoader(dataset, batch_size=hp_config["batch_size"], shuffle=True, sampler=sampler)

    # Total metric value across all batches
    euclidean_distance_total = 0

    # Total loss value across all batches
    loss_total = 0

    # Diagonal metric 
    d_metric = 0

    with torch.set_grad_enabled(phase == "train"):

        for  sample in dataloader:

            # Get the data from the sample
            if hp_config["model_type"] == "geddnet":
                leye  = sample["leye"].to(device)
                reye  = sample["reye"].to(device)
                face  = sample["face"].to(device)
                subj_id = torch.FloatTensor([1]).to(device)
                optimizer_ad_vgg.zero_grad()
                optimizer_ad_cls.zero_grad()
                y_hat, t_hat = model(face.float().to(device), leye.float().to(device), reye.float().to(device), subj_id.float().to(device))

            y = sample["y"].to(device)
            height_tensor = sample["screen_height_mm"].to(device)
            width_tensor =  sample["screen_width_mm"].to(device)
            
            # Zero the gradients before training
            if optimizer_ad:
                optimizer_ad.zero_grad()

            if (y_hat.shape != y.shape):
                logger_loss.warn("The shape of the y hat is %s and the y is %s" % (y_hat.shape, y.shape))

            #Compute the MSE loss
            loss = criterion(y.flatten().float(), y_hat.flatten().float().to(device))

            # Sum to total loss value
            loss_total += float(loss.item())*y.shape[0]

            # Compute the euclidean distance between the predicted and the actual gaze coordinates
            euclidean_distance_total += model_train_helper.evaluation_metric(y, y_hat, height_tensor, width_tensor)

            #Compute the diagonal metric 
            d_metric += model_train_helper.diagonal_dist_metric(y, y_hat, height_tensor, width_tensor)

            if phase == "train":
                
                if hp_config["use_l1_reg"]:
                    l1_pen = hp_config["l1_reg_lambda"]*sum([p.abs().sum() for p in model.parameters()])
                    loss += l1_pen
                elif  hp_config["use_l2_reg"]:
                    l2_pen = hp_config["l2_reg_lambda"]*sum([(p**2).sum() for p in model.parameters()])
                    
                    if hp_config["model_type"] == "geddnet":
                        l2_pen = 0
                        for k, v in model_named_params.items():
                            if "bias" in k:
                                continue
                            if any(str in k for str in ["1_1", "1_2", "2_1", "2_2"]):
                                l2_pen +=  0.01*(v**2).sum()
                            else:
                                l2_pen += (v**2).sum()
                        l2_pen*= hp_config["l2_reg_lambda"]

                    loss += l2_pen
                
                # Accumulate the gradients
                loss.backward()

                # Perform gradient descent
                if hp_config["model_type"] == "geddnet":
                    if epoch > hp_config["warm_up_epochs"]:
                        optimizer_ad_vgg.step()
                    optimizer_ad_cls.step()

            elif phase == "test":
                pass
            
    euclidean_distance_total /= len(dataset)
    loss_total /= len(dataset)
    d_metric = ((d_metric)/(len(dataset)))*100


    if log_loss:
        logger_loss.info("The total loss for phase: %s , epoch: %d, fold_num:%s is: %s " % 
        (phase, epoch, fold_num, loss_total))
    
    if log_evaluation_metric:
        logger_metric.info("The total euclidean distance for phase: %s , epoch: %d, fold_num:%s, is: %s " % 
        (phase, epoch, fold_num, euclidean_distance_total))
        logger_dmetric.info("The total diagonal metric for phase: %s , epoch: %d, fold_num:%s, is: %s " % 
        (phase, epoch, fold_num, d_metric))

    return loss_total, euclidean_distance_total, d_metric

# %%
def perform_validation_per_participant(epoch):
        loss_train, euclidean_distance_train, d_metric_train = execute_step(dataset_train,  "train", epoch,  0,  log_loss=False)

        # Testing 
        # Note that for printing metric across the folds set the log_evaluation_metric to True
        loss_test, euclidean_distance_test, d_metric_test = execute_step(dataset_test, "test", epoch, 0, log_evaluation_metric=False)

        return loss_train, euclidean_distance_train, d_metric_train, loss_test, euclidean_distance_test,d_metric_test

# %%
def iterate_and_train_model(**kwargs):

    train_loss_list = []
    test_loss_list = []
    train_eval_metric_list = []
    test_eval_metric_list = []
    train_d_metric_list = []
    test_d_metric_list = []
    
    best_loss = 1234567
    num_epochs_since_imp = 0
        

    # Iterating over the epochs
    for epoch in range(max(max(hp_config["min_epochs"], hp_config["num_epochs"]), min(hp_config["max_epochs"], hp_config["num_epochs"]))):
        
        if hp_config["k_fold"] > 1:
            # Perform k fold validation
            pass
        else:
            # Perform validation per participant
            loss_train_total,euclidean_distance_train_total, d_metric_train_total,  loss_test_total,euclidean_distance_test_total, d_metric_test_total = perform_validation_per_participant(epoch)

        if epoch > hp_config["plot_epoch_start"]:
            train_loss_list.append(loss_train_total)
            test_loss_list.append(loss_test_total)
            train_eval_metric_list.append(euclidean_distance_train_total)
            test_eval_metric_list.append(euclidean_distance_test_total)
            train_d_metric_list.append(d_metric_train_total)
            test_d_metric_list.append(d_metric_test_total)

        with open(str(model_train_path_dict["loss_file_path"]), 'a') as f:
            logger_loss.info(f"The loss diff is {(best_loss - loss_train_total)}")
            
        if (best_loss - loss_train_total) > hp_config["imp_thresh"]:
            num_epochs_since_imp = 0
            best_loss = loss_train_total
        else:
            num_epochs_since_imp += 1
                    
        # Save the model after every epoch
        
        if epoch%hp_config["num_epochs_to_save_model"] == 0:
            model_train_helper.save_model(model, model_train_path_dict["checkpoint_op_path"]  /  f"gaze_pred_{epoch}.pt")
        elif epoch == hp_config["num_epochs"] - 1:
            model_train_helper.save_model(model, model_train_path_dict["checkpoint_op_path"]  /  f"gaze_pred_{epoch}.pt")
        elif loss_train_total <= best_loss:
            model_train_helper.save_model(model, model_train_path_dict["checkpoint_op_path"] / "best_model.pt")


        # Print the average loss and euclidean distance
        if (epoch%hp_config["num_epochs_to_log_loss"]==0):
            logger_loss.info(f"The average  train loss(MSE for x,y co-ordinates) for epoch: {epoch} is: {loss_train_total}")
            logger_loss.info(f"The average  test loss(MSE for x,y co-ordinates) for epoch: {epoch} is: {loss_test_total}")

        if (epoch%hp_config["num_epochs_to_log_metric"]==0):
            logger_metric.info(f"The average euclidean distance train for epoch: {epoch} is: {euclidean_distance_train_total} mm")
            logger_metric.info(f"The average euclidean distance test  for epoch: {epoch} is: {euclidean_distance_test_total} mm")
            logger_dmetric.info(f"The average diagonal metric train for epoch: {epoch} is: {d_metric_train_total}")
            logger_dmetric.info(f"The average diagonal metric test for epoch: {epoch} is: {d_metric_test_total}")

        # Early stopping conditions
        if num_epochs_since_imp > hp_config["patience"]:
            hp_config["num_epochs"] = epoch + 1
            with open(str(model_train_path_dict["loss_file_path"]), 'a') as f:
                logger_loss.info("Early stopping after %d epochs." % (epoch))
            
            # We run one more iteration for gathering performance details
            if hp_config["k_fold"] > 1:
                # Perform k fold validation
                pass
            else:
                # Perform validation per participant
                loss_train_total,euclidean_distance_train_total, d_metric_train_total, loss_test_total,euclidean_distance_test_total, d_metric_test_total = perform_validation_per_participant(epoch)
            break

    # Plot the loss vs epochs
    model_train_helper.plot_train_test_metric_vs_epochs(train_loss_list, test_loss_list, 
    metric="Loss(MSE for x,y co-ordinates)", figure_save_path=model_train_path_dict["figures_op_path"] / "Loss_Plot" , title="Loss_Plot")
    model_train_helper.plot_train_test_metric_vs_epochs(train_eval_metric_list, test_eval_metric_list, 
    metric="Euclidean Distance in mm", figure_save_path=model_train_path_dict["figures_op_path"] / "Euclidean_Distance_Plot" , title="Euclidean_Distance_Plot")
    model_train_helper.plot_train_test_metric_vs_epochs(train_d_metric_list, test_d_metric_list, 
    metric="Diagonal Metric", figure_save_path=model_train_path_dict["figures_op_path"] / "Diagonal_Metric",  title="Diagonal_Metric")

# %%
def evaluate_model(dataset):

    dataloader = DataLoader(dataset, batch_size=hp_config["batch_size"])

    all_gs = []
    all_preds = []
    euc_errors = [[[] for j in range(11)] for i in range(11)]

    for sample in dataloader:

        # Get the data from the sample
        if hp_config["model_type"] == "geddnet":
            leye  = sample["leye"].to(device)
            reye  = sample["reye"].to(device)
            face  = sample["face"].to(device)
            subj_id = torch.FloatTensor([1]).to(device)
            y_hat, t_hat = model(face.float().to(device), leye.float().to(device), reye.float().to(device), subj_id.float().to(device))



        height_tensor = sample["screen_height_mm"].to(device)
        width_tensor =  sample["screen_width_mm"].to(device)
        y = sample["y"].to(device)
        all_gs.extend(y.tolist())
        all_preds.extend(y_hat.tolist())

        for i in range(len(y)):
            eu_val = model_train_helper.evaluation_metric(y[i][None], y_hat[i][None], 
            height_tensor[i][None], width_tensor[i][None])
                        
            # Storing the gold standard co-orindate vs [Eucd Errors]
            xx = math.floor(y[i][0]*10)
            yy = math.floor(y[i][1]*10)
            if xx >= 0 and xx <= 10 and  yy >=0 and yy <= 10:
                euc_errors[xx][yy].append(eu_val)

    return all_gs, all_preds, euc_errors

# %%
def plot_heatmaps():

    all_gs_train, all_preds_train, euc_errors_train = evaluate_model(dataset_train)
    all_gs_test, all_preds_test, euc_errors_test = evaluate_model(dataset_test)

    all_gs_test_np = np.array(all_gs_test)
    all_gs_train_np = np.array(all_gs_train)
    all_preds_train_np = np.array(all_preds_train)
    all_preds_test_np = np.array(all_preds_test)
    all_gs_np = np.concatenate((all_gs_train_np, all_gs_test_np))
    all_preds_np = np.concatenate((all_preds_train_np, all_preds_test_np))
    euc_errors_train_np = np.zeros((10, 10))
    euc_errors_test_np = np.zeros((10, 10))

    all_gs_test_bins = np.zeros((10, 10))
    all_gs_train_bins = np.zeros((10, 10))
    all_preds_train_bins = np.zeros((10, 10))
    all_preds_test_bins = np.zeros((10, 10))
    all_gs_bins = np.zeros((10, 10))
    all_preds_bins = np.zeros((10, 10))

    columns = [round(val*0.1, 1) for val in range(10)]

    model_train_helper.populate_bins(all_gs_train_np, all_gs_train_bins)
    model_train_helper.plot_heatmap(all_gs_train_bins, columns, "gold_standard_Train", 
                                figure_save_path=model_train_path_dict["figures_op_path"] / "gold_standard_Train")


    model_train_helper.populate_bins(all_gs_test_np, all_gs_test_bins)
    model_train_helper.plot_heatmap(all_gs_test_bins, columns, "gold_standard_Test", 
            figure_save_path=model_train_path_dict["figures_op_path"] / "gold_standard_Test")


    model_train_helper.populate_bins(all_preds_train_np, all_preds_train_bins)
    model_train_helper.plot_heatmap(all_preds_train_bins, columns, "Prediction_Train", 
        figure_save_path=model_train_path_dict["figures_op_path"] / "Prediction_Train")

    model_train_helper.populate_bins(all_preds_test_np, all_preds_test_bins)
    model_train_helper.plot_heatmap(all_preds_test_bins, columns, "Prediction_Test", 
        figure_save_path=model_train_path_dict["figures_op_path"] / "Prediction_Test")

    model_train_helper.populate_bins(all_preds_np, all_preds_bins)
    model_train_helper.plot_heatmap(all_preds_bins, columns, "All_Prediction", 
        figure_save_path=model_train_path_dict["figures_op_path"] / "All_Prediction")


    model_train_helper.populate_bins(all_gs_np, all_gs_bins)
    model_train_helper.plot_heatmap(all_gs_bins, columns, "All_Gold_Standards", 
        figure_save_path=model_train_path_dict["figures_op_path"] / "All_Gold_Standards")

    euc_errors_all = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if not (euc_errors_train[i][j] or euc_errors_test[i][j]):
                euc_errors_all[j][i] = None
            else:
                euc_errors_all[j][i] = round(np.mean(np.concatenate((euc_errors_train[i][j], euc_errors_test[i][j]))), 1)

    for i in range(10):
        for j in range(10):
            if euc_errors_train[i][j]:
                euc_errors_train_np[j][i] = round(np.mean( euc_errors_train[i][j]), 1)
            else:
                euc_errors_train_np[j][i] = None

    for i in range(10):
        for j in range(10):
            if euc_errors_test[i][j]:
                euc_errors_test_np[j][i] = round(np.mean( euc_errors_test[i][j]), 1)
            else:
                euc_errors_test_np[j][i] = None

    
    model_train_helper.plot_heatmap(euc_errors_train_np, columns, "Euclidean_Error_Train", 
        figure_save_path=model_train_path_dict["figures_op_path"] / "Euclidean_Error_Train")
    model_train_helper.plot_heatmap(euc_errors_test_np, columns, "Euclidean_Error_Test", 
        figure_save_path=model_train_path_dict["figures_op_path"] / "Euclidean_Error_Test")
    model_train_helper.plot_heatmap(euc_errors_all, columns, "Euclidean_Error_All", 
        figure_save_path=model_train_path_dict["figures_op_path"] / "Euclidean_Error_All")


# %%


model_path_utils = ModelPathUtils()
project_root_path = get_project_root()
hp_config_path = project_root_path / "src" / "webcam_eyetracking"/ "models" / 'model_training_config' / 'hyperparams.json'
with open(hp_config_path) as json_file:
    hp_config = json.load(json_file)


model_train_helper = ModelTrainingHelper(hp_config)
device = model_train_helper.get_training_device(2)

# %%
experiment_name = f"{hp_config['experiment_name']}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
model_train_path_dict = model_path_utils.create_or_update_model_train_paths(experiment_name)

logging_helper = LoggingHelper()
logger_loss= logging_helper.get_logger(f"{__name__}_loss_log")
logging_helper.add_handler(logger_loss, file_path=model_train_path_dict["loss_file_path"])
logger_metric= logging_helper.get_logger(f"{__name__}_metric_log")
logging_helper.add_handler(logger_metric, file_path=model_train_path_dict["metric_file_path"])
logger_dmetric = logging_helper.get_logger(f"{__name__}_diag_metric")
logging_helper.add_handler(logger_dmetric, file_path=model_train_path_dict["diag_metric_file_path"])


# %%
processed_dataset_folder = model_path_utils.get_processed_dataset_path()

vgg16_path = None
if hp_config["model_type"] == "geddnet":
    vgg16_path = processed_dataset_folder / "Model_Weights" / "vgg16_weights.npz"
model = model_train_helper.init_model(vgg16_path=vgg16_path)
model = model.to(device)

# MSE Los is used here since it's quite similar to the euclidean norm 
criterion = nn.MSELoss()

# Rprop optimizer is used for gradient descent. 
if hp_config["model_type"] == "geddnet":
    model_named_params = dict(model.named_parameters())
    model_cls_params = []
    model_vgg_params = []
    for k, v in model_named_params.items():
        if any(str in k for str in ["1_1", "1_2", "2_1", "2_2"]):
            model_vgg_params.append(v)
        else:
            model_cls_params.append(v)
    optimizer_ad = None    
    optimizer_ad_cls = optim.Rprop(model_cls_params, hp_config["learning_rate"])
    optimizer_ad_vgg = optim.Rprop(model_vgg_params, hp_config["learning_rate"])


# %%
dataset_train = torch.load(processed_dataset_folder / hp_config["dataset_folder"] / hp_config["dataset_train_file"])
dataset_test = torch.load(processed_dataset_folder / hp_config["dataset_folder"] / hp_config["dataset_test_file"])


# %%
with open(model_train_path_dict["exp_op_path"] / "hyperparams.json", "w") as json_file:
    json_file.write(json.dumps(hp_config, indent=2))

iterate_and_train_model()
plot_heatmaps()


