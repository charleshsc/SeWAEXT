import numpy as np
import torch

import time
import os 

from tqdm import trange

def save_checkpoint(state,name):
  filename =name
  torch.save(state, filename)

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, writer=None, variant=None, device='cuda'):
        self.model = model
        self.swa_model = None
        self.ema_model = None
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.swa_scheduler = None
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.writer = writer
        self.variant = variant
        self.device = device

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, logger=None, save_path=None, k=1):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for i in trange(num_steps, desc=f"Train Epoch {iter_num}", leave=False):
            train_loss = self.train_step()
            train_losses.append(train_loss)

            if self.ema_model is not None:
                self.ema_model.update(self.model)

            if self.ema_model is not None and (i+1) % k == 0:
                self.ema_model.apply(self.model)

            if self.swa_model is not None and (i+1) % k ==0:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                # self.model.load_state_dict(self.swa_model.state_dict(), strict=False)
            elif self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            if self.swa_model is not None:
                outputs = eval_fn(self.model, self.swa_model)
            else:
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

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
