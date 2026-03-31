import torch
import torch.nn as nn


class AllLoss:
    def __init__(self, loss_type):
        self.loss_type = loss_type
        self.eps = 1e-3  # For Charbonnier loss
        self.deta = 1  # For Huber loss

        # Instantiate the loss classes for MAE and MSE
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        self.mae_loss_fn = nn.L1Loss(reduction='mean')

    def charbonnier_loss(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss

    def huber_loss(self, X, Y):
        error = torch.abs(torch.add(X, -Y))
        cond = error <= self.deta
        loss_p = torch.where(cond, 0.5 * (error ** 2), self.deta * (error - 0.5 * self.deta))
        loss = torch.mean(loss_p)
        return loss

    def mse_loss(self, X, Y):
        # 使用 torch.nn.MSELoss 计算 MSE
        loss = self.mse_loss_fn(X, Y)
        return loss

    def mae_loss(self, X, Y):
        # 使用 torch.nn.L1Loss 计算 MAE
        loss = self.mae_loss_fn(X, Y)
        return loss

    def __call__(self, X, Y):
        if self.loss_type == 'MAE':
            return self.mae_loss(X, Y)
        elif self.loss_type == 'MSE':
            return self.mse_loss(X, Y)
        elif self.loss_type == 'CB':  # Charbonnier
            return self.charbonnier_loss(X, Y)
        elif self.loss_type == 'HB':  # Huber
            return self.huber_loss(X, Y)
        else:
            raise ValueError("Invalid loss type. Choose from 'MAE', 'MSE', 'CB', 'HB'.")
