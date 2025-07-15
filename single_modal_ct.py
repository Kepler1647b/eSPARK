# k-fold
import os
import numpy as np
import pandas as pd

from mymodels import CLAM_SB_multimodal2, CLAM_SB_multimodal2_patho, CLAM_SB_multimodal2_ct
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

import torchmetrics
# import logger from lightning
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from tqdm import tqdm
from einops import repeat

import timm.scheduler
# import multiprocessing


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

topkn = 50
ct_root = "/home/21/zihan/Storage/ESO/ct_feat_radfm/"
wsi_root = "/data15/data15_5/dexia/eso/celltype_feat"
hilbert_path = "/data15/data15_5/dexia/eso/ESO/patho_processing/feat_uni_256/"
label_path = "/data15/data15_5/dexia/eso/"

preget_omics_feature = "/home/21/zihan/Storage/ESO/code_2409/dexia_ct_feat_1205.npy"
preget_omics_feature = np.load(preget_omics_feature, allow_pickle=True).tolist()

omics_root = "/home/21/zihan/Storage/ESO/ct_feat_omic"
omics_dict_path = "/home/21/zihan/Storage/ESO/ctfeat_lasso_1124.npy"
omics_type_list = ["bw5/tumor", "bw5/eso"]
lasso = False
omics_dict = np.load(omics_dict_path, allow_pickle=True).item()

symptom_root = '/data15/data15_5/dexia/eso/celltype_feat'
symptoms = ["tumor necrosis", "tumor budding", "immune cells infiltrating the tumor stroma", "well-differentiated tumor cells"]


true_case_list_sysucc, true_rad_list_sysucc = [], []
true_case_list_henan, true_rad_list_henan = [], []
true_case_list_shantou, true_rad_list_shantou = [], []

for dataset_flag in ["sysucc", "henan", "shantou"]:
    cur_wsi_path = os.path.join(wsi_root, dataset_flag)
    cur_case_list = []
    for file in os.listdir(cur_wsi_path):
        if file.endswith("_x.pt") or file.endswith("_y.pt"):
            continue
        cur_case_list.append(file.split(".")[0])
    cur_case_list = sorted(cur_case_list)
    print("Number of cases in {}: {}".format(dataset_flag, len(cur_case_list)))

    if dataset_flag == "sysucc":
        true_case_list_sysucc = cur_case_list
    elif dataset_flag == "henan":
        true_case_list_henan = cur_case_list
    elif dataset_flag == "shantou":
        true_case_list_shantou = cur_case_list

for dataset_flag in ["sysucc", "henan", "shantou"]:
    cur_ct_path = os.path.join(ct_root, dataset_flag)
    cur_case_list = os.listdir(cur_ct_path)
    # split("+").[0]
    cur_case_list = [case.split("+")[0] for case in cur_case_list]
    cur_case_list = sorted(cur_case_list)
    print("Number of cases in {}: {}".format(dataset_flag, len(cur_case_list)))

    if dataset_flag == "sysucc":
        true_rad_list_sysucc = cur_case_list
    elif dataset_flag == "henan":
        true_rad_list_henan = cur_case_list
    elif dataset_flag == "shantou":
        true_rad_list_shantou = cur_case_list

true_case_list_dict = {
    "sysucc": true_case_list_sysucc,
    "henan": true_case_list_henan,
    "shantou": true_case_list_shantou
}
true_rad_list_dict = {
    "sysucc": true_rad_list_sysucc,
    "henan": true_rad_list_henan,
    "shantou": true_rad_list_shantou
}


# dataset has the input of ct_root and label_path and the casepatho_id that in the ct_root
class CustomDataset(Dataset):
    def __init__(self, case_list=None, dataset_flag="sysucc", mean=None, std=None):

        ct_path_ = os.path.join(ct_root, dataset_flag)
        wsi_path_ = os.path.join(wsi_root, dataset_flag)
        hilbert_path_ = os.path.join(hilbert_path, dataset_flag)
        label_path_ = os.path.join(label_path, "label_" + dataset_flag + ".csv")
        symptom_root_ = os.path.join(symptom_root, dataset_flag)

        self.ct_root = ct_path_
        self.wsi_root = wsi_path_
        self.hilbert_path = hilbert_path_
        self.label_path = label_path_

        self.data = []
        self.data_omics = []
        self.data_hilbert = []
        self.data_ct = []
        self.data_ct_x = []
        self.data_case = []
        self.labels = []

        true_case_list = true_case_list_dict[dataset_flag]
        true_rad_list = true_rad_list_dict[dataset_flag]

        # for all omics type, read and concat to form a unified omics data
        omics_data_list = []
        for omics_type in omics_type_list:
            omics_path = os.path.join(omics_root, omics_type, "ESO-Radiomics-features_{}_ns1024.xlsx".format(dataset_flag))
            # omics_path = os.path.join(omics_root, omics_type, "ESO-Radiomics-features_{}.xlsx".format(dataset_flag))
            omics_df = pd.read_excel(omics_path)
            if lasso:
                # get the columns to be used
                omics_type_str = omics_type.replace("/", "_")
                omics_type_str = omics_type.split('/')[-1]
                omics_columns = omics_dict["ind_" + omics_type_str]
                # add the first column to the columns
                omics_columns = np.insert(omics_columns, 0, "Unnamed: 0")
                # print("omics_columns: ", omics_columns)

                omics_df = omics_df[omics_columns]
            omics_df.columns = ["Unnamed: 0"] + [omics_type + "_" + col for col in omics_df.columns[1:]]
            omics_data_list.append(omics_df)
        
        Unnamed_0 = omics_data_list[0]["Unnamed: 0"]
        omics_df = pd.concat(omics_data_list, axis=1)
        # omics_df drop ["Unnamed: 0"] column
        omics_df = omics_df.drop(columns=["Unnamed: 0"])
        omics_df.insert(0, "Unnamed: 0", Unnamed_0)

        # get the label
        label_df = pd.read_csv(self.label_path)
        # label_df = label_df[["patho_id", "label", "rad_id"]]
        if dataset_flag == "sysucc":
            label_df = label_df[["patho_id", "label", "rad_id", "input_format"]]
        else:
            label_df = label_df[["patho_id", "label", "rad_id"]]
        # drop the rows that have NaN values
        label_df = label_df.dropna()
        # to string
        label_df["patho_id"] = label_df["patho_id"].astype(str)
        if dataset_flag == "henan":
            label_df["rad_id"] = label_df["rad_id"].astype(int)
        label_df["rad_id"] = label_df["rad_id"].astype(str)
        # label_df["exam_time"] = label_df["exam_time"].astype(str)
        # drop the rows that are not in the true_case_list
        label_df = label_df[label_df["patho_id"].isin(true_case_list)]
        # drop the rows that are not in the true_rad_list
        label_df = label_df[label_df["rad_id"].isin(true_rad_list)]

        if case_list is None:
            # get the case_list from label_df
            case_list = label_df["patho_id"].unique()
        
        # sort
        case_list = sorted(case_list)
        self.case_list = case_list

        print("dataset_flag: ", dataset_flag)
        print("Number of cases: ", len(case_list))

        print(len(case_list))

        self.preget_omics_feature = []
        # add the preget_omics_feature to self.preget_omics_feature along the case_list
        for case in case_list:
            # print("case: ", case)
            # print("preget_omics_feature[case]: ", preget_omics_feature[case][0])
            # raise ValueError("stop here")
            self.preget_omics_feature.append(preget_omics_feature[case][0].cpu().numpy())
        # to tensor
        self.preget_omics_feature = torch.tensor(self.preget_omics_feature, dtype=torch.float32)
        print("self.preget_omics_feature: ", self.preget_omics_feature.shape)
        # raise ValueError("stop here")
            

        for case in case_list:
            # if sysucc, get the input_format from label_df
            if dataset_flag == "sysucc":
                input_format = label_df[label_df["patho_id"] == case]["input_format"].values[0]
            else:
                input_format = label_df[label_df["patho_id"] == case]["rad_id"].values[0]
            if dataset_flag == "sysucc":
                rad_id_index = omics_df[omics_df["Unnamed: 0"] == str(input_format)].index
            else:
                rad_id_index = omics_df[omics_df["Unnamed: 0"] == int(input_format)].index
            # print("rad_id_index: ", rad_id_index)
            if len(rad_id_index) == 0:
                raise ValueError("rad_id not found in omics_df")
                
            rad_id_index = rad_id_index[0]
            omics_data = omics_df.iloc[rad_id_index][1:]
            # print("omics_data: ", omics_data)
            omics_data = omics_data.to_numpy().astype(np.float32)
            self.data_omics.append(omics_data)

        # to tensor
        self.data_omics = torch.tensor(self.data_omics, dtype=torch.float32)
        # do standardization on the omics data along the columns
        if mean is None or std is None:
            mean = self.data_omics.mean(dim=0)
            std = self.data_omics.std(dim=0)
        self.mean = mean
        self.std = std
        eps = 1e-16
        # eps = 0

        self.data_omics = (self.data_omics - mean) / (std + eps)

        print("self.data_omics: ", self.data_omics.shape)
        # raise ValueError("stop here")

        # load symptom data
        for case in tqdm(case_list, total=len(case_list), colour="yellow"):
            # get the data
            #data_path = os.path.join(symptom_root_, case)
            data_path = os.path.join(symptom_root_)

            #eso_tumor = torch.load(os.path.join(data_path, "squamous cell carcinoma.pt"), map_location=torch.device("cpu"))
            eso_tumor = torch.load(os.path.join(data_path, "%s.pt" % case), map_location=torch.device("cpu"))

            # concat
            # data = torch.cat((tumor_necrosis, tumor_budding, immune_cells_infiltrating_the_tumor_stroma, well_differentiated_tumor_cells), dim=0)
            data = eso_tumor
            self.data.append(data)
            self.data_case.append(case)
            # get the label
            label = label_df[label_df["patho_id"] == case]["label"].values[0]
            self.labels.append(label)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data_omics[idx], self.labels[idx], self.case_list[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, dim, depth, dropout, num_classes, activation="sigmoid"):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, dim))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(dim, dim))
            self.layers.append(nn.BatchNorm1d(dim))
            self.layers.append(eval("nn." + activation)())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(dim, num_classes))

        # initialize the weights
        self.apply(initialize_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# do k-fold
# model hyperparameters
model_hyperparameters = {
    "k": 10,
    "num_classes": 2,
    "input_dim": 192 if lasso else 2164 * len(omics_type_list),
    "dim": 192,
    "depth": 3,
    "activation": "Identity",
    "dropout": 0.,
    "labelsmoothing": 0.2,
    "seed": 3407,
    "batch_size": 256,
    "num_workers": 0,
    "prefetch_factor": 2,
    "num_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "gradient_accumulation_steps": 1
}
gradient_accumulation_steps = model_hyperparameters["gradient_accumulation_steps"]
comment = "single_modal_ct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(model_hyperparameters["seed"])

# load the label
sysucc_label_df = None
for dataset_flag in ["sysucc", "henan", "shantou"]:
    print("data source: ", dataset_flag)
    cur_label_path = os.path.join(label_path, "label_" + dataset_flag + ".csv")
    label_df = pd.read_csv(cur_label_path)
    label_df = label_df[["patho_id", "label", "rad_id"]]
    # drop the rows that have NaN values
    label_df = label_df.dropna()
    print("Number of cases without NaN values: ", len(label_df["patho_id"].unique()))
    # to string
    label_df["patho_id"] = label_df["patho_id"].astype(str)
    if dataset_flag == "henan":
        label_df["rad_id"] = label_df["rad_id"].astype(int)
    label_df["rad_id"] = label_df["rad_id"].astype(str)
    # label_df["exam_time"] = label_df["exam_time"].astype(str)

    true_case_list = true_case_list_dict[dataset_flag]
    true_rad_list = true_rad_list_dict[dataset_flag]
    # drop the rows that are not in the true_case_list
    label_df_anti1 = label_df[~label_df["patho_id"].isin(true_case_list)]
    label_df_anti2 = label_df[~label_df["rad_id"].isin(true_rad_list)]
    label_df = label_df[label_df["patho_id"].isin(true_case_list)]
    
    # drop the rows that are not in the true_rad_list
    # print("true_rad_list: ", true_rad_list)
    label_df = label_df[label_df["rad_id"].isin(true_rad_list)]
    

    # print information
    print("Number of cases: ", len(label_df["patho_id"].unique()))
    print("Number of slides: ", len(label_df))
    print("Number of label: ", label_df["label"].value_counts())

    if sysucc_label_df is None and dataset_flag == "sysucc":
        sysucc_label_df = label_df

# label ratio
label_ratio = sysucc_label_df["label"].value_counts().values

fold_file_path = "/data15/data15_5/dexia/eso/train_valid_cases_fold_4.npy"
train_valid_cases = np.load(fold_file_path, allow_pickle=True).item()
# the file contains the train and valid cases and idx
# {
#     "train_cases": train_cases,
#     "valid_cases": valid_cases,
#     "train_idx": train_idx,
#     "valid_idx": valid_idx
# }
train_cases_all = train_valid_cases["train_cases"]
test_inner_cases = train_valid_cases["valid_cases"]
train_label = []
for case in train_cases_all:
    train_label.append(sysucc_label_df[sysucc_label_df["patho_id"] == case]["label"].values[0])
test_inner_label = []
for case in test_inner_cases:
    test_inner_label.append(sysucc_label_df[sysucc_label_df["patho_id"] == case]["label"].values[0])


skf = StratifiedKFold(n_splits=model_hyperparameters["k"], shuffle=True, random_state=model_hyperparameters["seed"])
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_cases_all, train_label)):
    print("Fold: ", fold)

    train_cases = np.array(train_cases_all)[train_idx]
    valid_cases = np.array(train_cases_all)[valid_idx]

    train_dataset = CustomDataset(train_cases, "sysucc")
    train_mean = train_dataset.mean
    train_std = train_dataset.std
    valid_dataset = CustomDataset(valid_cases, "sysucc", train_mean, train_std)
    test_dataset = CustomDataset(test_inner_cases, "sysucc", train_mean, train_std)
    henan_dataset = CustomDataset(None, "henan", train_mean, train_std)
    shantou_dataset = CustomDataset(None, "shantou", train_mean, train_std)

    train_loader = DataLoaderX(train_dataset, 
                               batch_size=model_hyperparameters["batch_size"], 
                               shuffle=True, 
                               num_workers=model_hyperparameters["num_workers"],
                               pin_memory=True)
    valid_loader = DataLoaderX(valid_dataset, 
                               batch_size=model_hyperparameters["batch_size"], 
                               shuffle=False, 
                               num_workers=model_hyperparameters["num_workers"],
                               pin_memory=True)
    test_inner_loader = DataLoaderX(test_dataset,
                                    batch_size=model_hyperparameters["batch_size"],
                                    shuffle=False,
                                    num_workers=model_hyperparameters["num_workers"],
                                    pin_memory=True)
    henan_loader = DataLoaderX(henan_dataset, 
                               batch_size=model_hyperparameters["batch_size"], 
                               shuffle=False, 
                               num_workers=model_hyperparameters["num_workers"],
                               pin_memory=True)
    shantou_loader = DataLoaderX(shantou_dataset, 
                               batch_size=model_hyperparameters["batch_size"], 
                               shuffle=False, 
                               num_workers=model_hyperparameters["num_workers"],
                               pin_memory=True)


    model = MLP(
        input_dim=model_hyperparameters["input_dim"],
        dim=model_hyperparameters["dim"],
        depth=model_hyperparameters["depth"],
        dropout=model_hyperparameters["dropout"],
        num_classes=model_hyperparameters["num_classes"],
        activation=model_hyperparameters["activation"]
    )

    model.to(device)

    # print number of parameters
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", num_params)

    # get weight for the loss function according to the label ratio
    weight = torch.tensor([label_ratio[1] / label_ratio[0], 1.0], device=device, dtype=torch.float32)

    # criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=model_hyperparameters["labelsmoothing"])
    criterion = nn.CrossEntropyLoss(label_smoothing=model_hyperparameters["labelsmoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=model_hyperparameters["lr"], weight_decay=model_hyperparameters["weight_decay"])
    scheduler = timm.scheduler.CosineLRScheduler(optimizer,
                                                 t_initial=model_hyperparameters["num_epochs"],
                                                 lr_min=1e-6,
                                                 warmup_lr_init=0, 
                                                 warmup_t=5)

    # do training, use torchmetrics to record the metrics, and use tensorboard to log the training process
    logger = TensorBoardLogger(comment, name="tensorboard_fold_{}".format(fold))
    # create a log file to record the training process
    csv_logger = CSVLogger(comment, name="CSV_fold_{}".format(fold))

    metric_data_list = ["train", "valid", "test", "henan", "shantou"]
    metric_names = ["Accuracy", "AUC"]

    metrics_dict = {}
    for metric_data in metric_data_list:
        metrics_dict[metric_data + "_Loss"] = torchmetrics.MeanMetric().to(device)
        metrics_dict[metric_data + "_Accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
        metrics_dict[metric_data + "_AUC"] = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)
    
    test_loader_list = [valid_loader, test_inner_loader, henan_loader, shantou_loader]
    logger.log_graph(model, torch.randn(1, 1, 2048).to(device))

    # log the hyperparameters
    logger.log_hyperparams(model_hyperparameters)
    csv_logger.log_hyperparams(model_hyperparameters)

    for epoch in tqdm(range(model_hyperparameters["num_epochs"]), colour="green"):

        scheduler.step(epoch)
        model.train()
        tmp_cnt = 0
        for data_idx, (data_omics, label, data_case) in tqdm(enumerate(train_loader), total=len(train_loader), colour="blue"):
            data_omics = data_omics.to(device)
            label = label.to(device)

            output = model(data_omics)
            output_softmax = F.softmax(output, dim=1)

            loss = criterion(output, label)

            # gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (data_idx + 1) % gradient_accumulation_steps == 0 or data_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            # record the metrics
            # for metric in train_metrics:
            #     metric(output_softmax.cpu(), label.cpu())
            metrics_dict["train_Loss"].update(loss)
            for metric_name in metric_names:
                metrics_dict["train_" + metric_name].update(output_softmax.cpu(), label.cpu())
        
        str_metrics = ""
        for metric_name in metric_names:
            str_metrics += "{}: {:.4f}, ".format(metric_name, metrics_dict["train_" + metric_name].compute())
            # metrics_dict["train_" + metric_name].reset()

        # print(f"Epoch: {epoch}, Loss: {train_loss / len(train_loader)}, Accuracy: {train_metrics[0].compute()}, Precision: {train_metrics[1].compute()}, Recall: {train_metrics[2].compute()}, AUC: {train_metrics[3].compute()}")
        print(f"Epoch: {epoch}, Loss: {metrics_dict['train_Loss'].compute()}, {str_metrics}")

        model.eval()
        all_auc = 0
        labellist = []
        valuelist = []
        tylist = []
        namelist = []
        iiii = 0
        with torch.no_grad():
            # for test_loader, test_metrics, test_name in zip(test_loader_list, test_metrics_list, test_name_list):
            for test_loader, test_name in zip(test_loader_list, metric_data_list[1:]):
                iiii+=1
                for data_omics, label, data_case in tqdm(test_loader, total=len(test_loader), colour="red"):
                    data_omics = data_omics.to(device)
                    label = label.to(device)

                    output = model(data_omics)
                    output_softmax = F.softmax(output, dim=1)
                    loss = criterion(output, label)
                    for i in range(len(label)):
                        labellist.append(label.cpu().numpy()[i])
                        valuelist.append(output_softmax.cpu().numpy()[i][1])
                        namelist.append(data_case[i])
                        tylist.append(iiii)

                    metrics_dict[test_name + "_Loss"].update(loss)
                    for metric_name in metric_names:
                        metrics_dict[test_name + "_" + metric_name].update(output_softmax.cpu(), label.cpu())

                str_metrics = ""
                for metric_name in metric_names:
                    str_metrics += "{}: {:.4f}, ".format(metric_name, metrics_dict[test_name + "_" + metric_name].compute())
                    # metrics_dict[test_name + "_" + metric_name].reset()

                print(f"Name: {test_name}, Epoch: {epoch}, Loss: {metrics_dict[test_name + '_Loss'].compute()}, {str_metrics}")

                # print AUC
                if test_name != "train" and test_name != "valid":
                    # print("AUC: ", metrics_dict[test_name + "_AUC"].compute())
                    all_auc += metrics_dict[test_name + "_AUC"].compute()
        csv_path = os.path.join(comment, str(fold))
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        csv_result = pd.DataFrame({'dataset': tylist, 'name': namelist, 'value': valuelist, 'label': labellist})
        csv_result.to_csv(os.path.join(csv_path, '%s.csv' % epoch))

        metric_dict = {}
        
        # out loop: type of metrics, including AUC, Accuracy, Precision, Recall
        # in loop: type of data, including valid, henan, shantou
        for metric_name in metric_names:

            for metric_data in metric_data_list:
                metric_dict[metric_data + "_" + metric_name] = metrics_dict[metric_data + "_" + metric_name].compute()
                metrics_dict[metric_data + "_" + metric_name].reset()


        logger.log_metrics(metric_dict, step=epoch)
        csv_logger.log_metrics(metric_dict, step=epoch)

        train_loss = 0
        valid_loss = 0
        
        # # save the log
        logger.save()
        csv_logger.save()