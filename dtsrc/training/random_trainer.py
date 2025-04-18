import numpy as np
import torch
import copy
import time
import os 
import gc
import random


from tqdm import trange
from dtsrc.training.merge_utils import MergeNet

def save_checkpoint(state,name):
  filename =name
  torch.save(state, filename)

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, writer=None, variant=None, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.writer = writer
        self.variant = variant
        self.device = device

        self.state_dict_list = []

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, logger=None, save_path=None, k=1):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for i in trange(num_steps, desc=f"Train Epoch {iter_num}", leave=False):
            train_loss = self.train_step(self.model, self.optimizer)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            self.state_dict_list.append(self.model.state_dict())
            if len(self.state_dict_list) == self.variant['merge_number']:
                self.merge_model(self.variant['merge_steps'], iter_num, i, logger)

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.log('=' * 80)
        logger.log(f'Iteration {iter_num}')
        best_ret = -10000
        best_nor = 0
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'normalized_score' in k:
                best_nor = max(best_nor, float(v))
            logger.log(f'{k}: {v}')

        logs['Best_return_mean'] = best_ret

        if self.writer is not None:
            self.writer.add_scalar("Train/Loss", np.mean(train_losses), iter_num)
            self.writer.add_scalar("Val/Return_mean", best_ret, iter_num)
            self.writer.add_scalar("Val/Normalized_Return_mean", best_nor, iter_num)

        return logs
    
    def merge_model(self, num_steps, iter_num, step, logger):
        eval_logits = []
        train_losses = []

        ada_model = MergeNet(copy.deepcopy(self.model), self.state_dict_list, temperature=self.variant['t'], k=self.variant['merge_k'], device=self.device).to(self.device)
        
        random_numbers = random.sample(range(len(self.state_dict_list)), self.variant['merge_k'])
        for i in random_numbers:
            ada_model.mask_logit.data[i] = 1.
            
        ada_model.get_model()
        infer_model = ada_model.infer_model.train()
        for p in infer_model.parameters():
            p.requires_grad = True
        self.model.load_state_dict(infer_model.state_dict())
        self.optimizer.state_dict()['state'].clear()  # 清空优化器的状态字典
        self.state_dict_list = []
        del ada_model
        gc.collect()
        torch.cuda.empty_cache()
    
    def eval_iteration(self, num_steps, iter_num=0, logger=None, save_path=None):

        logs = dict()

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        # logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        # logs['training/train_loss_mean'] = np.mean(train_losses)
        # logs['training/train_loss_std'] = np.std(train_losses)

        # for k in self.diagnostics:
        #     logs[k] = self.diagnostics[k]

        logger.log('=' * 80)
        logger.log(f'Iteration {iter_num}')
        best_ret = -10000
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            logger.log(f'{k}: {v}')

        logs['Best_return_mean'] = best_ret

        return logs

    def train_step(self, model, optimizer):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
        optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
