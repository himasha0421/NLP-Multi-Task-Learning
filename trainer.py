# Built-in libraries
import copy
import datetime
from typing import Dict, List
# Third-party libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
# Local files
from utils import save
from config import LABEL_DICT

class Trainer():
    '''
    The trainer for training models.
    It can be used for both single and multi task training.
    Every class function ends with _m is for multi-task training.
    '''
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        loss_weights: List[float],
        clip: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        patience: int,
        task_name: str,
        model_name: str,
        seed: int
    ):
        self.model = model
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.clip = clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.task_name = task_name
        self.model_name = model_name
        self.seed = seed
        self.datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')

        # Evaluation results
        self.train_losses = []
        self.test_losses = []
        self.valid_losses = []
        
        self.train_f1 = []
        self.test_f1 = []
        self.valid_f1 = []
        
        self.best_train_f1 = 0.0
        self.best_test_f1 = 0.0
        self.best_valid_f1 = 0.0
        
        # Evaluation results for multi-task
        self.best_train_f1_m = np.array([0, 0, 0, 0, 0], dtype=np.float64)
        self.best_test_f1_m = np.array([0, 0, 0 ,0 ,0], dtype=np.float64)
        self.best_valid_f1_m = np.array([0, 0, 0 ,0 ,0], dtype=np.float64)

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch()
            self.test()
            print(f'Best test f1: {self.best_test_f1:.4f}')
            print('=' * 20)

        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1, self.best_test_f1),
            f'./save/results/single_{self.task_name}_{self.datetimestr}_{self.best_test_f1:.4f}.pt'
        )

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Training'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                loss += _loss.item()
                y_pred = logits.argmax(dim=1).cpu().numpy()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append(f1)
        if f1 > self.best_train_f1:
            self.best_train_f1 = f1

    def test(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Testing'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                loss += _loss.item()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append(f1)
        if f1 > self.best_test_f1:
            self.best_test_f1 = f1
            self.save_model()

    def train_m(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch_m()
            # validation after each epoch
            self.test_m( 'valid' )
            
            print(f'Best test results sentiment_score: {self.best_test_f1_m[0]:.4f}')
            print(f'Best test results annotator_score: {self.best_test_f1_m[1]:.4f}')
            print(f'Best test results directness_score: {self.best_test_f1_m[2]:.4f}')
            print(f'Best test results group_score: {self.best_test_f1_m[3]:.4f}')
            print(f'Best test results target_score: {self.best_test_f1_m[4]:.4f}')
            print('=' * 20)

        print('Saving results ...')
        save(
            ( self.train_losses , self.test_losses , self.train_f1 , self.test_f1 , self.best_train_f1_m , self.best_test_f1_m ),
            f'./save/results/mtl_{self.datetimestr}_{self.best_test_f1_m[0]:.4f}.pt' )

    def train_one_epoch_m(self):
        self.model.train()
        dataloader = self.dataloaders['train']

        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        y_pred_all_D = None
        y_pred_all_E = None
        
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None
        labels_all_D = None
        labels_all_E = None

        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, label_A, label_B, label_C , label_D , label_E in tqdm(dataloader, desc='Training M'):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)
            label_D = label_D.to(device=self.device)
            label_E = label_E.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                # logits_A, logits_B, logits_C = self.model(inputs, mask)
                all_logits = self.model(inputs, lens, mask)
                
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1].argmax(dim=1).cpu().numpy()
                y_pred_C = all_logits[2].argmax(dim=1).cpu().numpy()
                y_pred_D = all_logits[3].argmax(dim=1).cpu().numpy()
                y_pred_E = all_logits[4].argmax(dim=1).cpu().numpy()


                labels_all_A = label_A.cpu().numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.cpu().numpy()))
                labels_all_B = label_B.cpu().numpy() if labels_all_B is None else np.concatenate((labels_all_B, label_B.cpu().numpy()))
                labels_all_C = label_C.cpu().numpy() if labels_all_C is None else np.concatenate((labels_all_C, label_C.cpu().numpy()))
                labels_all_D = label_D.cpu().numpy() if labels_all_D is None else np.concatenate((labels_all_D, label_D.cpu().numpy()))
                labels_all_E = label_E.cpu().numpy() if labels_all_E is None else np.concatenate((labels_all_E, label_E.cpu().numpy()))

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = y_pred_B if y_pred_all_B is None else np.concatenate((y_pred_all_B, y_pred_B))
                y_pred_all_C = y_pred_C if y_pred_all_C is None else np.concatenate((y_pred_all_C, y_pred_C))
                y_pred_all_D = y_pred_D if y_pred_all_D is None else np.concatenate((y_pred_all_D, y_pred_D))
                y_pred_all_E = y_pred_E if y_pred_all_E is None else np.concatenate((y_pred_all_E, y_pred_E))

                # f1[0] += self.calc_f1(label_A, y_pred_A)
                # f1[1] += self.calc_f1(Non_null_label_B, Non_null_pred_B)
                # f1[2] += self.calc_f1(Non_null_label_C, Non_null_pred_C)

                _loss = self.loss_weights[0] * self.criterion(all_logits[0], label_A)
                _loss += self.loss_weights[1] * self.criterion(all_logits[1], label_B)
                _loss += self.loss_weights[2] * self.criterion(all_logits[2], label_C)
                _loss += self.loss_weights[3] * self.criterion(all_logits[3], label_D)
                _loss += self.loss_weights[4] * self.criterion(all_logits[4], label_E)
                
                loss += _loss.item()

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        # compute f1 scores
        f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
        f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
        f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')
        f1_D = f1_score(labels_all_D, y_pred_all_D, average='macro')
        f1_E = f1_score(labels_all_E, y_pred_all_E, average='macro')

        print(f'loss = {loss:.4f}')
        print(f' sentiment_score: {f1_A:.4f}')
        print(f' annotator_score: {f1_B:.4f}')
        print(f' directness_score: {f1_C:.4f}')
        print(f' group_score: {f1_D:.4f}')
        print(f' target_score: {f1_E:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append([f1_A, f1_B, f1_C , f1_D , f1_E ])

        if f1_A > self.best_train_f1_m[0]:
            self.best_train_f1_m[0] = f1_A
        if f1_B > self.best_train_f1_m[1]:
            self.best_train_f1_m[1] = f1_B
        if f1_C > self.best_train_f1_m[2]:
            self.best_train_f1_m[2] = f1_C
        if f1_D > self.best_train_f1_m[3]:
            self.best_train_f1_m[3] = f1_D
        if f1_E > self.best_train_f1_m[4]:
            self.best_train_f1_m[4] = f1_E
    # ------------------------------  need to modify the test , valid functions and the mtl model 
    def test_m(self , stage='test' ):
        self.model.eval()
        dataloader = self.dataloaders[ stage ]
        loss = 0
        iters_per_epoch = 0
        
        # define the placeholders for predictions and labels
        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        y_pred_all_D = None
        y_pred_all_E = None
        
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None
        labels_all_D = None
        labels_all_E = None

        for inputs, lens, mask, label_A, label_B, label_C , label_D , label_E in tqdm(dataloader, desc='{} M'.format(stage) ):
            
            iters_per_epoch += 1

            labels_all_A = label_A.numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.numpy()))
            labels_all_B = label_B.numpy() if labels_all_B is None else np.concatenate((labels_all_B, label_B.numpy()))
            labels_all_C = label_C.numpy() if labels_all_C is None else np.concatenate((labels_all_C, label_C.numpy()))
            labels_all_D = label_D.numpy() if labels_all_D is None else np.concatenate((labels_all_D, label_D.numpy()))
            labels_all_E = label_E.numpy() if labels_all_E is None else np.concatenate((labels_all_E, label_E.numpy()))

            # define the input ids , mask and sentence lens
            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            
            # define the labels for each use case
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)
            label_D = label_D.to(device=self.device)
            label_E = label_E.to(device=self.device)

            with torch.set_grad_enabled(False):
                # infer the model
                all_logits = self.model(inputs, lens, mask)
                # define the model predictions
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1].argmax(dim=1).cpu().numpy()
                y_pred_C = all_logits[2].argmax(dim=1).cpu().numpy()
                y_pred_D = all_logits[3].argmax(dim=1).cpu().numpy()
                y_pred_E = all_logits[4].argmax(dim=1).cpu().numpy()

                # f1[0] += self.calc_f1(label_A, y_pred_A)
                # f1[1] += self.calc_f1(label_B, y_pred_B)
                # f1[2] += self.calc_f1(label_C, y_pred_C)

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = y_pred_B if y_pred_all_B is None else np.concatenate((y_pred_all_B, y_pred_B))
                y_pred_all_C = y_pred_C if y_pred_all_C is None else np.concatenate((y_pred_all_C, y_pred_C))
                y_pred_all_D = y_pred_D if y_pred_all_D is None else np.concatenate((y_pred_all_D, y_pred_D))
                y_pred_all_E = y_pred_E if y_pred_all_E is None else np.concatenate((y_pred_all_E, y_pred_E))

                _loss =  self.loss_weights[0] * self.criterion(all_logits[0], label_A)
                _loss += self.loss_weights[1] * self.criterion(all_logits[1], label_B)
                _loss += self.loss_weights[2] * self.criterion(all_logits[2], label_C)
                _loss += self.loss_weights[3] * self.criterion(all_logits[3], label_D)
                _loss += self.loss_weights[4] * self.criterion(all_logits[4], label_E)
                
                # get the total loss
                loss += _loss.item()

        loss /= iters_per_epoch
        
        if( stage == 'valid'  ):
            # calculate the f1-score value
            f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
            f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
            f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')
            f1_D = f1_score(labels_all_D, y_pred_all_D, average='macro')
            f1_E = f1_score(labels_all_E, y_pred_all_E, average='macro')

            print(f' loss = {loss:.4f}')
            print(f' F1 sentiment_score: {f1_A:.4f}')
            print(f' F1 annotator_score: {f1_B:.4f}')
            print(f' F1 directness_score: {f1_C:.4f}')
            print(f' F1 group_score: {f1_D:.4f}')
            print(f' F1 target_score: {f1_E:.4f}')

            self.test_losses.append(loss)
            self.test_f1.append([ f1_A, f1_B, f1_C , f1_D , f1_E ])

            if f1_A > self.best_test_f1_m[0]:
                self.best_test_f1_m[0] = f1_A
                self.save_model()

            if f1_B > self.best_test_f1_m[1]:
                self.best_test_f1_m[1] = f1_B
            if f1_C > self.best_test_f1_m[2]:
                self.best_test_f1_m[2] = f1_C
            if f1_D > self.best_test_f1_m[3]:
                self.best_test_f1_m[3] = f1_D
            if f1_E > self.best_test_f1_m[4]:
                self.best_test_f1_m[4] = f1_E
                
        else:
            return (labels_all_A, y_pred_all_A) , (labels_all_B, y_pred_all_B) , (labels_all_C, y_pred_all_C) , (labels_all_D, y_pred_all_D) ,(labels_all_E, y_pred_all_E)

        #  for i in range(len(f1)):
        #     for j in range(len(f1[0])):
        #         if f1[i][j] > self.best_test_f1_m[i][j]:
        #             self.best_test_f1_m[i][j] = f1[i][j]
        #             if i == 0 and j == 0:
        #                 self.save_model()

    def calc_f1(self, labels, y_pred):
        return np.array([
            f1_score(labels.cpu(), y_pred.cpu(), average='macro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='micro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='weighted')
        ], np.float64)

    def printing(self, loss, f1):
        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1[0]:.4f}')
        # print(f'Micro-F1 = {f1[1]:.4f}')
        # print(f'Weighted-F1 = {f1[2]:.4f}')

    def save_model(self):
        print('Saving model...')
        if self.task_name == 'all':
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1_m[0]}_seed{self.seed}.pt'
        else:
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1}_seed{self.seed}.pt'
        save(copy.deepcopy(self.model.state_dict()), filename)
