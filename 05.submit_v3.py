

#--
import sys
#sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/07.Challenge/01.MapYourCity_HuggingFace")
import map_dataset
import map_train
#from models import *

#sys.path.append("/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/00.Libs")
#import RS_dataset
#import RS_models
import RS_utils
#--- torch
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
#--- loss functions
from utils.losses import LabelSmoothCrossEntropy, CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
#---
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.classification import Accuracy
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import argparse
import yaml 
import timm
import numpy as np 
import time
import wandb
#from rich.console import Console
from wandb.integration.lightning.fabric import WandbLogger
from tqdm import tqdm 

device = "cuda:7" 
batch_size = 8

#--- all infos
inference_dict ={
    'models':[],
    'folds' :[],
    'data' :[],
    'cfgs':[],
    'predictions':[],
    'labels':[]
}

#--- argparser
cfgs_names = ['finetune_35.yaml', 'finetune_36.yaml','finetune_37.yaml']
for cfg_name in cfgs_names:    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=os.path.join('/root/configs', cfg_name))
    args = parser.parse_args(args=[])
    cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))
    
    for fold_ in range(cfg.N_SPLIT):        
        inference_dict['cfgs'].append(cfg)
        inference_dict['folds'].append(fold_)
        inference_dict['data'].append(cfg.DATA_TYPE)
        print("Model run version : ", cfg.RUN_VERSION)
        print("Model run fold : ", fold_)
        print("Data type : ", cfg.DATA_TYPE)


#--- Data 
input_path = "/data/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_data = sorted( os.listdir(train_path) )

names_label = []
for ID in names_data:
    y = int(open(train_path + ID + '/label.txt', "r").read())
    names_label.append(y)


for cfg in inference_dict['cfgs']:   
    model = timm.create_model(
    cfg.MODEL,
    pretrained=True,
    num_classes=cfg.CLASSES_NUM )

    #--- data config and transform
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=False)

    inference_dict['models'].append(model)
    print("#------------------------------------")
    print(" Model Name : ",cfg.MODEL)
    
    


#--- all the candidates for ensembles based on validation score 

saved_root = "/data/eric"
check_points = sorted(os.listdir(saved_root))
target_runs = list(set([ cfg.RUN_VERSION for cfg in inference_dict['cfgs']]))



def find_best_model(target_run_n):
    
    target_runs_0 = [i for i in check_points if str(i.split("_")[0]) == str(target_run_n) ]
    
    best_model_runs = []
    for fold_n in range(0,cfg.N_SPLIT):
        fold_s = [ i for i in target_runs_0 if str(i.split("_")[-5]) == str(fold_n) ]
        #print(fold_s)
        best_model = ""
        best_score = 0
        for fq in fold_s:
            score =  float(fq.split("_")[-3])
            if score > best_score:
                score = best_score
                best_model = fq
        best_model_runs.append(best_model)
    return best_model_runs 


#--- find all 
global_best_models = []
for tg in target_runs:
    fold_best_models = find_best_model(tg)
    global_best_models.extend(fold_best_models)


# Define the categories in the desired order
categories = ['streetview', 'topview', 'sentinel2']

# Create a dictionary to hold lists of file paths for each category
categorized_files = {category: [] for category in categories}

# Categorize the file paths
for path in global_best_models:
    for category in categories:
        if category in path:
            categorized_files[category].append(path)
            break

# Reorder the file paths based on the desired order
ordered_file_paths = []
for category in categories:
    ordered_file_paths.extend(categorized_files[category])

# Print the ordered file paths
for path in ordered_file_paths:
    print(path)

# load weights
ckpt_paths =[ os.path.join(saved_root,i) for i in ordered_file_paths]
for i,model in enumerate(inference_dict['models']):
    model.load_state_dict(torch.load(ckpt_paths[i])['model'] )
    
    
valid_set = map_dataset.Map_Dataset_v14(names_data,train_path,max_size=data_config['input_size'][1],cfg=cfg,split="valid")  

#--- Data 
input_path = "/data/"
train_path = input_path + "train/data/"
test_path = input_path + "test/data/"
train_df = pd.read_csv(input_path + "train/train-set.csv")
test_df = pd.read_csv(input_path + "test/test-set.csv") 

#--- data split 
names_test = os.listdir(test_path)
names_test = sorted(names_test)


submit_df = pd.DataFrame(
    {"idx":[i for i in range(len(names_test))],
     "names_test":names_test,
     "street_view_exists":[False for i in range(len(names_test))],
     "model_0_street_view_prediction" :[[] for i in range(len(names_test))],
     "model_1_street_view_prediction" :[[] for i in range(len(names_test))],
     "model_2_street_view_prediction" :[[] for i in range(len(names_test))],
     "model_3_street_view_prediction" :[[] for i in range(len(names_test))],
     "model_4_street_view_prediction" :[[] for i in range(len(names_test))],

     "model_5_topview_prediction" :[[] for i in range(len(names_test))],
     "model_6_topview_prediction" :[[] for i in range(len(names_test))],
     "model_7_topview_prediction" :[[] for i in range(len(names_test))],
     "model_8_topview_prediction" :[[] for i in range(len(names_test))],
     "model_9_topview_prediction" :[[] for i in range(len(names_test))],

     "model_10_sentinelview_prediction" :[[] for i in range(len(names_test))],
     "model_11_sentinelview_prediction" :[[] for i in range(len(names_test))],
     "model_12_sentinelview_prediction" :[[] for i in range(len(names_test))],
     "model_13_sentinelview_prediction" :[[] for i in range(len(names_test))],
     "model_14_sentinelview_prediction" :[[] for i in range(len(names_test))]
     })



print(submit_df.head())

for ID in names_test:
    street_file_ = os.path.join( test_path + ID + '/street.jpg')
    if os.path.exists(street_file_):
        submit_df.loc[submit_df['names_test'] == ID, 'street_view_exists'] = True
        
print(submit_df.head())


street_view_names = submit_df.loc[submit_df['street_view_exists'] == True, 'names_test']
street_view_names = sorted(street_view_names)


#-----------------------------------


for model_idx in range(5):
        
    #--------------------
    if model_idx < 5:
        view_name = "street_view"
    elif 5 <= model_idx < 10:
        view_name = "topview"
    elif 10 <= model_idx:
        view_name = "sentinelview"


    #--------------------
    model = inference_dict['models'][model_idx]
    cfg = inference_dict['cfgs'][model_idx]

    def find_data_config(cfg):
        model = timm.create_model(
        cfg.MODEL,
        pretrained=True,
        num_classes=cfg.CLASSES_NUM )

        #--- data config and transform
        data_config = timm.data.resolve_model_data_config(model)

        return data_config
    #---- 
    data_config = find_data_config(cfg)
    test_set = map_dataset.Map_Dataset_v14(street_view_names,test_path,max_size=data_config['input_size'][1],cfg=cfg,split="test") 
    TestLoader = DataLoader(test_set,batch_size,shuffle=False)


    #-- Loader train/valid
    Loader = TestLoader

    predictions_ = []
    model = model.to(device) 
    for batch in tqdm(Loader):
        #print("model idx : ", model_idx)
        if model_idx < 5:
            input_img = batch[0].to(device)
        elif 5 <= model_idx < 10:
            input_img = batch[1].to(device)
        elif 10 <= model_idx:
            input_img = batch[2].to(device)

        batch_preds = model(input_img)
        #-- 
        predictions_.extend(batch_preds.detach().cpu())

    #--- insert prediction into DataFrame 
    cnt = 0
    for name, pred in tqdm(zip(street_view_names,predictions_)):
        #print(name,pred)
        #pred_ = [i.softmax(-1).argmax(-1).numpy() for i in predictions_][cnt]
        pred_ = [i.softmax(-1).numpy() for i in predictions_][cnt]
        #print(pred_)
        #submit_df.loc[submit_df['names_test'] == name, 'street_view_prediction']._append(pd.DataFrame( np.asarray(pred)) )
        
        Obj = submit_df['names_test'] == name
        idx_true = [ (i,v) for (i,v) in Obj.items() if v == True][0][0]
        #submit_df.loc[idx_true, 'street_view_exists'] = pred_
        submit_df.at[idx_true, f'model_{model_idx}_{view_name}_prediction'] = pred_
        cnt +=1

    #--- check
    #cnt == len(street_view_names)
    
print(submit_df)


print("running")

infer_names = [i for i in submit_df['names_test']]
for model_idx in range(5,10):

    #--------------------
    if model_idx < 5:
        view_name = "street_view"
    elif 5 <= model_idx < 10:
        view_name = "topview"
    elif 10 <= model_idx:
        view_name = "sentinelview"


    #--------------------
    model = inference_dict['models'][model_idx]
    cfg = inference_dict['cfgs'][model_idx]

    def find_data_config(cfg):
        model = timm.create_model(
        cfg.MODEL,
        pretrained=True,
        num_classes=cfg.CLASSES_NUM )

        #--- data config and transform
        data_config = timm.data.resolve_model_data_config(model)

        return data_config
    #---- 
    data_config = find_data_config(cfg)


    #---
    test_set = map_dataset.Map_Dataset_v14(infer_names,test_path,max_size=data_config['input_size'][1],cfg=cfg,split="test",test_mode="top_view_only") 
    TestLoader = DataLoader(test_set,batch_size,shuffle=False)


    #-- Loader train/valid
    Loader = TestLoader

    predictions_ = []
    model = model.to(device) 
    for batch in tqdm(Loader):
        #print("model idx : ", model_idx)
        if model_idx < 5:
            input_img = batch[0].to(device)
        elif 5 <= model_idx < 10:
            input_img = batch[1].to(device)
        elif 10 <= model_idx:
            input_img = batch[2].to(device)

        batch_preds = model(input_img)
        #-- 
        predictions_.extend(batch_preds.detach().cpu())

    #--- insert prediction into DataFrame 
    cnt = 0
    for name, pred in tqdm(zip(infer_names,predictions_)):
        pred_ = [i.softmax(-1).numpy() for i in predictions_][cnt]

        submit_df.at[cnt, f'model_{model_idx}_{view_name}_prediction'] = pred_
        cnt +=1

    #--- check
    #cnt == len(street_view_names)
    

infer_names = [i for i in submit_df['names_test']]
for model_idx in range(10,15):

    #--------------------
    if model_idx < 5:
        view_name = "street_view"
    elif 5 <= model_idx < 10:
        view_name = "topview"
    elif 10 <= model_idx:
        view_name = "sentinelview"


    #--------------------
    model = inference_dict['models'][model_idx]
    cfg = inference_dict['cfgs'][model_idx]

    def find_data_config(cfg):
        model = timm.create_model(
        cfg.MODEL,
        pretrained=True,
        num_classes=cfg.CLASSES_NUM )

        #--- data config and transform
        data_config = timm.data.resolve_model_data_config(model)

        return data_config
    #---- 
    data_config = find_data_config(cfg)


    #---
    test_set = map_dataset.Map_Dataset_v14(infer_names,test_path,max_size=data_config['input_size'][1],cfg=cfg,split="test",test_mode="top_view_only") 
    TestLoader = DataLoader(test_set,batch_size,shuffle=False)

    #-- Loader train/valid
    Loader = TestLoader

    predictions_ = []
    model = model.to(device) 
    for batch in tqdm(Loader):
        #print("model idx : ", model_idx)
        if model_idx < 5:
            input_img = batch[0].to(device)
        elif 5 <= model_idx < 10:
            input_img = batch[1].to(device)
        elif 10 <= model_idx:
            input_img = batch[2].to(device)

        batch_preds = model(input_img)
        #-- 
        predictions_.extend(batch_preds.detach().cpu())

    #--- insert prediction into DataFrame 
    cnt = 0
    for name, pred in tqdm(zip(infer_names,predictions_)):
        pred_ = [i.softmax(-1).numpy() for i in predictions_][cnt]

        submit_df.at[cnt, f'model_{model_idx}_{view_name}_prediction'] = pred_
        cnt +=1

    #--- check
    #cnt == len(street_view_names)

submit_df.to_csv("/data/eric_submission/large_ensemble_v2.1_batch_16_prob.csv",index=False)

#--------------------------------------------------
# Ensemble
#--------------------------------------------------

tmp_ = []
for i,row in submit_df.iterrows():
    #print(row['street_view_exists'])
    
    if row['street_view_exists'] == True:
        #print(row.values[3:])
        result_ = sum( row.values[3:] ) / len(row.values[3:]) 
        tmp_.append(result_)
    else:
        result_ = sum( row.values[8:] ) / len(row.values[8:]) 
        tmp_.append(result_)


sub_file = pd.read_csv("/data/test/test-set.csv")
sub_file.sort_values(by=["pid"], inplace=True)
sub_file['predicted_label'] = [ np.argmax(i) for i in tmp_]
sub_file.to_csv("/data/eric_submission/large_ensemble_v2.1_batch_16.csv",index=False)
