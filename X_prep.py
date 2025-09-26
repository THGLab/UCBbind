import random,os
import pandas as pd
from datetime import datetime
from dataset import CustomDataSet, collate_fn
from model import AttentionDTA
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from tensorboardX import SummaryWriter
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr

from torch.utils.data import DataLoader, Subset

def test_precess(model,pbar):
    loss_f = nn.MSELoss()
    model.eval()
    test_losses = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds,proteins, labels, smiles, sequences = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = labels.cuda()
            predicts= model.forward(compounds,  proteins)
            loss = loss_f(predicts, labels.view(-1, 1))
            total_preds = torch.cat((total_preds, predicts.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)
            test_losses.append(loss.item())
    Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
    test_loss = np.average(test_losses)
    rmse = np.sqrt(mean_squared_error(Y, P))
    pearson, _ = pearsonr(Y, P)
    return Y, P, test_loss, mean_squared_error(Y, P), mean_absolute_error(Y, P), r2_score(Y, P), rmse, pearson

def test_model(test_dataset_load,save_path,DATASET,lable = "Train",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(test_dataset_load)),
        total=len(test_dataset_load))
    T, P, loss_test, mse_test, mae_test, r2_test, rmse_test, pearson_test = test_precess(model,test_pbar)
    if save:
        with open(save_path + "/{}_stable_{}_prediction.txt".format(DATASET,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};MSE:{:.5f};RMSE{:.5f};MAE:{:.5f};Pearson:{:.5f};R2:{:.5f}.' \
        .format(lable, loss_test, mse_test, rmse_test, mae_test, pearson_test, r2_test)
    print(results)
    return results,mse_test, mae_test, r2_test, rmse_test, pearson_test

def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    """select seed"""
    SEED = 4321
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """Load preprocessed data."""
    #DATASET = "Moonshot"
    # DATASET = "PDBbind"
    DATASET = "BindingDB"

    tst_path = f'datasets/{DATASET}.csv'
    df = pd.read_csv(tst_path)  
    cpi_list = df.apply(lambda row: f" {row['SMILES']} {row['Sequence']} {row['Value']}", axis=1).tolist()

    print("load finished")
    print("data shuffle")
    dataset = shuffle_dataset(cpi_list, SEED)
    K_Fold = 1
    Batch_size = 128
    weight_decay = 1e-4
    Learning_rate = 5e-5
    Patience = 50
    Epoch = 500
    """Output files."""
    save_path = "./Results/{}/".format(DATASET)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + '{}_results.txt'.format(DATASET)

    MSE_List, MAE_List, R2_List, RMSE_List, Pearson_List = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'Using fixed splits', '*' * 25)
        train_indices = df.index[df['Split'] == 'Train'].tolist()
        val_indices = df.index[df['Split'] == 'Val'].tolist()
        test_indices = df.index[df['Split'] == 'Test'].tolist()

        trainset = torch.utils.data.Subset(dataset, train_indices)
        validset = torch.utils.data.Subset(dataset, val_indices)
        testset = torch.utils.data.Subset(dataset, test_indices)

        train_dataset_load = DataLoader(trainset, batch_size=Batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dataset_load = DataLoader(validset, batch_size=Batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataset_load = DataLoader(testset, batch_size=Batch_size, shuffle=False, collate_fn=collate_fn)

        train_size = len(trainset)

        """ create model"""
        model = AttentionDTA().cuda()
        """weight initialize"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        LOSS_F = nn.MSELoss()
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=Learning_rate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=Learning_rate, max_lr=Learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=train_size // Batch_size)
        save_path = "module_x/training/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        note = ""
        writer = SummaryWriter(log_dir=save_path, comment=note)

        """Start training."""
        print('Training...')
        start = timeit.default_timer()
        patience = 0
        best_score = 100
        for epoch in range(1, Epoch + 1):
            trian_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_dataset_load)),
                total=len(train_dataset_load))
            """train"""
            train_losses_in_epoch = []
            model.train()
            for trian_i, train_data in trian_pbar:
                '''data preparation '''
                trian_compounds, trian_proteins, trian_labels, train_smiles_batch, train_sequences_batch = train_data
                trian_compounds = trian_compounds.cuda()
                trian_proteins = trian_proteins.cuda()
                trian_labels = trian_labels.cuda()

                optimizer.zero_grad()

                predicts = model.forward(trian_compounds, trian_proteins)
                train_loss = LOSS_F(predicts, trian_labels.view(-1, 1))
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()

                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)
            writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)

            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
            valid_losses_in_epoch = []
            model.eval()
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_proteins, valid_labels, valid_smiles_batch, valid_sequences_batch = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_labels = valid_labels.cuda()
                    valid_predictions = model.forward(valid_compounds, valid_proteins)
                    valid_loss = LOSS_F(valid_predictions, valid_labels.view(-1, 1))
                    valid_losses_in_epoch.append(valid_loss.item())
                    total_preds = torch.cat((total_preds, valid_predictions.cpu()), 0)
                    total_labels = torch.cat((total_labels, valid_labels.cpu()), 0)
                Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
            valid_MSE = mean_squared_error(Y, P)
            valid_MAE = mean_absolute_error(Y, P)
            valid_R2 = r2_score(Y, P)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            if valid_MSE < best_score:
                best_score = valid_MSE
                patience = 0
                torch.save(model.state_dict(), save_path + 'valid_best_checkpoint.pth')
            else:
                patience+=1
            epoch_len = len(str(Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_MSE: {valid_MSE:.5f} ' +
                         f'valid_MAE: {valid_MAE:.5f} ' +
                         f'valid_R2: {valid_R2:.5f} ')
            print(print_msg)
            writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid MSE', valid_MSE, epoch)
            writer.add_scalar('Valid MAE', valid_MAE, epoch)
            writer.add_scalar('Valid R2', valid_R2, epoch)

            if patience == Patience:
                break
        torch.save(model.state_dict(), save_path + 'stable_checkpoint.pth')
        """load trained model"""
        model.load_state_dict(torch.load(save_path + "valid_best_checkpoint.pth"))
        trainset_test_results,_,_,_,rmse_train, pearson_train = test_model(train_dataset_load, save_path, DATASET, lable="Train")
        validset_test_results,_,_,_,rmse_valid, pearson_valid = test_model(valid_dataset_load, save_path, DATASET, lable="Valid")
        with open(save_path + "The_results.txt", 'a') as f:
            f.write("results on {}th fold\n".format(i_fold+1))
            f.write(trainset_test_results + '\n')
            f.write(validset_test_results + '\n')
        writer.close()




