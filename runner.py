import torch
from torch import nn
import typing, time
import torch.nn.functional as F
from typing import List, Tuple


class Runner:
    '''
    Handles training, validation and testing. The current version is specifically for the 
    Character Aware NLM paper, but the idea is to keep train and validation functions as 
    general as possible and have the user override the __call__ function to add additional 
    functionalities depending upon the paper.
    '''
    def __init__(self, model, optimizer, iterators, device, epochs):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.train_iterator = iterators['train_iterator']
        self.valid_iterator = iterators['valid_iterator']
        self.test_iterator = iterators['test_iterator']
        
    def train(self, iterator):
        '''
        Trains the language model with the iterator passed.
        '''
        
        print("Starting training....")
        self.model.train()
        train_losses = []
        train_ppl = []
        for batch_idx, batch in enumerate(iterator):
            
            if batch_idx % 300 == 0:
                print(f"Starting batch: {batch_idx}")
            x = batch['inputs'].to(self.device)
            y = batch['targets'].to(self.device)

            preds = self.model(x)

            loss = F.cross_entropy(preds, y.view(-1))

            train_ppl.append(torch.exp(loss.data))

            train_losses.append(loss.item())

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()

        train_losses = torch.tensor(train_losses,dtype=torch.float)
        train_ppl = torch.tensor(train_ppl,dtype=torch.float)
        
        return torch.mean(train_losses).item(), torch.mean(train_ppl).item()
        
    
    def validate(self, iterator):
        '''
        Validates the language model with the iterator passed.
        '''
        
        print("Starting validation....")
        self.model.eval()
        valid_losses = []
        valid_ppl = []

        for batch_idx, batch in enumerate(iterator):
            
            if batch_idx % 50 == 0:
                print(f"Starting batch: {batch_idx}")
            x = batch['inputs'].to(self.device)
            y = batch['targets'].to(self.device)
            
            with torch.no_grad():

                preds = self.model(x)

                loss = F.cross_entropy(preds, y.view(-1))

                valid_ppl.append(torch.exp(loss.data))

                valid_losses.append(loss.item())    

        valid_losses = torch.tensor(valid_losses,dtype=torch.float)
        valid_ppl = torch.tensor(valid_ppl,dtype=torch.float)
        
        return torch.mean(valid_losses).item(), torch.mean(valid_ppl).item()
    
    def test(self):
        test_loss, test_ppl = self.validate(self.train_iterator)
        
    def epoch_time(self, start_time, end_time):
        '''
        Helper function to record epoch time.
        '''
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def __call__(self):
        '''
        This is currently more specific to a particular paper, where the
        authors reduce the lr by half if the perplexity of the validation set
        does not decrease by more than 1.
        '''
        
        best_valid_ppl = 1000
        old_ppl = 1000
        
        for epoch in range(self.epochs):

            print(f"Epoch: {epoch+1}")

            start_time = time.time()

            train_loss, train_ppl = self.train(self.train_iterator)
            valid_loss, valid_ppl = self.validate(self.valid_iterator)

            old_lr = self.optimizer.param_groups[0]['lr']

            if valid_ppl < best_valid_ppl:
                best_valid_ppl = valid_ppl

            if (old_ppl - valid_ppl) <= 1.0:
                new_lr = old_lr/2
                self.optimizer.param_groups[0]['lr'] = new_lr
                print(f"Halving the learning rate from{old_lr} to {new_lr}")


            old_ppl = valid_ppl

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
            print(f"Epoch valid loss: {valid_loss}")
            print(f"Train PPL: {train_ppl}")
            print(f"Valid PPL: {valid_ppl}")
            print("====================================================================================")