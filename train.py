from __future__ import print_function
import argparse
import os
import sys
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

from dataset.rgb_dataset import TriDatasetRGB
from loss_metric import evaluate

from models.unet import UNet
from utils.tools import get_config


# training configurations
WEIGHTED_LOSS_ON = False
SAVE_EVERY_MODEL = False
MODEL_DIR = "./saved_models"

# Training settings
parser = argparse.ArgumentParser(description='Custom linear model')
parser.add_argument('--config', type=str, default='configs/unet_config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')


def train_model(model, module, num_epochs=5):
    
    since = time.time()

    best_model = model
    best_loss = sys.maxsize
    patience = 0
    train_loss = {}
    val_loss = {}
    
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.75)
    torch.autograd.set_detect_anomaly(True)
    
    early_stop = False
    
    # Training and val loop
    for epoch in range(num_epochs):
        if early_stop:
            break
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        if config["extract"]:
            phases = ['val']
        else:
            phases = ['train', 'val']
        for phase in phases:
#         for phase in ['val']:
            batch_loss = []
    
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
#             optimizer_g = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=1, gamma=0.5)
            # Training batch loop
            for i_batch, batch in enumerate(current_loader[phase]):
                
                landsats = batch["landsat"]
                sentinels = batch["sentinel"]
                cloudys = batch["cloudy"]
                targets = batch['target']
                masks = batch['mask']
                
                if cuda:
                    landsats = landsats.cuda()
                    sentinels = sentinels.cuda()
                    cloudys = cloudys.cuda()
                    targets = targets.cuda()
                    masks = masks.cuda()
                if config['extract']:
                    inpainted, *_ = model(landsats, sentinels, cloudys, targets, masks)
                        
                    outputs = inpainted.detach().cpu().numpy()
                    input_landsats = landsats.detach().cpu().numpy()
                    input_sentinels = sentinels.detach().cpu().numpy()
                    input_cloudys = cloudys.detach().cpu().numpy()
                    ground_truth = targets.detach().cpu().numpy()
                    
                    np.save(config["save_path"] + save_name + "/" + "results_outputs" + str(i_batch) + ".npy", outputs)
                    np.save(config["save_path"] + save_name + "/" + "results_landsats" + str(i_batch) + ".npy", input_landsats)
                    np.save(config["save_path"] + save_name + "/" + "results_sentinels" + str(i_batch) + ".npy", input_sentinels)
                    np.save(config["save_path"] + save_name + "/" + "results_cloudys" + str(i_batch) + ".npy", input_cloudys)
                    np.save(config["save_path"] + save_name + "/" + "results_ground_truth" + str(i_batch) + ".npy", ground_truth)
                    print("Saved sample ", i_batch)
                #print(features.shape, targets.shape, masks.shape)
                # outputs = model(features)
                #step_loss, inpainted, offset_flow = model(features, targets, masks)
                if config['selected_model'] == 'ada_conv':
                    inpainted, step_loss, fine_loss = model(landsats, sentinels, cloudys, targets, masks)
                    batch_loss.append(fine_loss.item())
                    running_loss += fine_loss.item()
                else:
                    inpainted, step_loss = model(landsats, sentinels, cloudys, targets, masks)
                    batch_loss.append(step_loss.item())
                    running_loss += step_loss.item()
                    
                if phase == 'train':
                    
                    optimizer_g.zero_grad()
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad:
                    #         print(name, param.data)
                    step_loss.backward()
                    # try:
                    #     step_loss.backward()
                    # except RuntimeError:
                    #     print(step_loss)
                    optimizer_g.step()
                    
                    if i_batch % 100 == 0:
                        print("batch_loss", i_batch, step_loss)
                else:
                    if i_batch % 1000 == 0:
                        print("batch_loss", i_batch, step_loss)
            
            if cuda:
                torch.cuda.empty_cache()
            if phase == 'train' and config["lr_decay"]:
                if optimizer_g.param_groups[0]["lr"] > 0.002:
                    scheduler.step()
            
            np.savetxt(config["save_path"] + "/" + save_name + "/" + "batch_loss_epoch_" + str(epoch) + "_" + phase + ".csv",
                       np.array(batch_loss), delimiter=",")
            epoch_loss = running_loss / len(current_set[phase])
            if phase == "train":
                train_loss[epoch] = epoch_loss
                print('\nEpoch: {} {} Loss: {:.6f}'.format(epoch + 1, phase, epoch_loss))
                print("Current learning rate,", optimizer_g.param_groups[0]["lr"])
#                 if epoch > 0 and epoch % 2 == 0:
#                 optimizer_g.param_groups[0]["lr"] *= 0.5 
            else:
                val_loss[epoch] = epoch_loss
                print('\nEpoch: {} {} Loss: {:.6f}'.format(epoch + 1, phase, epoch_loss))

            # if phase == 'val':
            time_elapsed = time.time() - since
            print('Time Elapsed :{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # Save best model
            if phase == 'val':
                if epoch_loss < best_loss or SAVE_EVERY_MODEL:
                    patience = 0
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    print('Best loss: {:6f}'.format(best_loss))
                    model_name = "_best_val_" + str(epoch) + ".pt"
                    torch.save(model.state_dict(), os.path.join(config["save_path"] + "/" + save_name, model_name))
                    print("Saved model :", model_name)
                else:
                    patience += 1
                    print("Loss did not improve, patience: " + str(patience))
                    print('Best loss: {:6f}'.format(best_loss))

            # Early stopping
            if patience > config["patience"] and config["early_stop"]:
                early_stop = True
                break

            # save training checkpoint
            # if phase == 'train':
            #     checkname = opt.save_path + "_epoch_{}.pth".format(epoch)
            #     torch.save({'epoch': epoch,
            #                 'state_dict': model.state_dict(),
            #                 'optimizer': optimizer.state_dict(),
            #                 }, checkname)
            #     print("Checkpoint saved to {}".format(checkname))

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    training_history = {}
    training_history["train_loss"] = train_loss
    training_history["val_loss"] = val_loss

    return best_model, training_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# adjust this function to implement sliding-window/rolling manner
def train_val_split(dataset, n_cv, val_size=6000):
    print("validation size,", val_size)
    # train_idx, val_idx = train_test_split(list(range(dataset.__len__())), test_size=val_split, shuffle=False)
    datasets = []
    for i in range(n_cv):
        datasets.append({})
        indices = list(range(dataset.__len__()))
        fold_len = len(indices) // n_cv
        train_idx = indices[:(fold_len * (i + 1) - val_size)]
        val_idx = indices[(fold_len * (i + 1) - val_size):(fold_len * (i + 1))]
        datasets[i]['train'] = Subset(dataset, train_idx)
        datasets[i]['val'] = Subset(dataset, val_idx)
    # datasets['train'] = dataset
    # datasets['val'] = dataset
    return datasets


if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.config)
    # substitute old exp_name
    save_name = "_".join([config['selected_model'], config["dataset"], str(config['nepoch']), str(config['batch_size']), str(config['lr']), config['expname']])
    print("running experiment", save_name)
    if config['selected_model'] == "edsr":
        model = ResModel(config)
    else:
        pass
    #model = model.double()
    path = config["save_path"] + "/" + save_name
    if os.path.isdir(path):
        print("experiment exists! Will overwrite existing models.")
    else:
        os.mkdir(path)
    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True
        torch.cuda.empty_cache()

    if config["dataset"] == "LS_3CFRGB":
        dataset = TriDatasetRGB(config["train_data_path"])
        
    if config["extract"]:
        subsets = train_val_split(dataset, config["num_cv"], 1000)
    else:
        subsets = train_val_split(dataset, config["num_cv"], config["val_size"])
    # save train and val loss for each epoch
    for cv_idx in range(config["num_cv"]):
        print("Start training cv set: " + str(cv_idx))
        current_set = subsets[cv_idx]
        current_loader = {'train': DataLoader(current_set['train'], batch_size=config["batch_size"], shuffle=False),
                          'val': DataLoader(current_set['val'], batch_size=1, shuffle=False)}

        # call model trainer here
        #model_ft = GeneratorOnly(config)
        if config["selected_model"] == "unet":
            model_ft = UNet(config)
      
        if cuda:
            model_ft = model_ft.to(config['gpu_ids'][0])
        #print("model details:", model_ft)
        if cuda:
            model_ft = nn.parallel.DataParallel(model_ft, device_ids=device_ids)
            model_ft_module = model_ft.module
        else:
            model_ft_module = model_ft

        if config["resume"]:
            checkpoint = torch.load(config["resume"])
            model_ft.load_state_dict(checkpoint, strict=False)
            print("loading existing models...")
        #print(model_ft_module)

        # criterion = evaluate.loss_mse()

        #optimizer_ft = optim.Adam(model_ft.parameters(), lr=config["lr"])

        # Run the functions and save the best model in the function model_ft.
        if config["extract"]:
            model_ft, training_history = train_model(model_ft, model_ft_module, num_epochs=1)
        else:
            model_ft, training_history = train_model(model_ft, model_ft_module, num_epochs=config["nepoch"])

        print("Training done")

        # Save model and history
        if not config["extract"]:
            model_name = save_name  + "_cv_" + str(cv_idx) + ".pt"
            torch.save(model_ft.state_dict(), os.path.join(path, model_name))
            print("Saved model :", model_name)

        # for _, key in enumerate(training_history["event_param"]):
        #     value = training_history["event_param"][key]
        #     np.save(MODEL_DIR + "/event_param_" + CODE + "_" + str(key) + ".npy", value)
        # rmspe_df = pd.DataFrame.from_dict(training_history['event_rmspe'], orient='index')
        # rmspe_df.to_csv(MODEL_DIR + "/event_rmspe_" + CODE + ".csv")
        if not config['extract']:
            rolling_keys = ["train_loss", "val_loss"]
            rolling_dict = {key: list(value.values()) for key, value in training_history.items() if key in rolling_keys}
            rolling_df = pd.DataFrame.from_dict(rolling_dict, orient='columns')
            idx = training_history[rolling_keys[0]].keys()
            rolling_df.index = list(idx)

            rolling_df.to_csv(config["save_path"] + "/" + save_name +"/" + "train_val_results_cv_" + str(cv_idx) + ".csv")
