import torch
import cv2

def log_metrics(metrics, step, tb_writer, wandb_writer):
    # Log metrics to TensorBoard
    if tb_writer:
        for key, value in metrics.items():
            tb_writer.add_scalar(key, value, step)
    
    # Log metrics to wandb
    # if wandb_writer:
    #     wandb_writer.log(metrics, step=step)


def td_lambda(rewards, predicted_discount, values, lambda_, device):
    """
    Compute the TD(λ) returns for value estimation.

    Args:
    - states (Tensor): Tensor of states with shape [batch_size, time_steps, state_dim].
    - rewards (Tensor): Tensor of rewards with shape [batch_size, time_steps].
    - predicted_discount (Tensor): Tensor indicating probability of episode termination with shape [batch_size, time_steps].
    - values (Tensor): Tensor of value estimates with shape [batch_size, time_steps].
    - lamda_ (float): The λ parameter in TD(λ) controlling bias-variance tradeoff.

    Returns:
    - td_lambda (Tensor): The computed lambda returns with shape [batch_size, time_steps - 1].
    """
    batch_size, seq_len, _ = rewards.shape
    last_lambda = torch.zeros((batch_size, 1)).to(device)
    cur_rewards = rewards[:, :-1]
    next_values = values[:, 1:]
    predicted_discount = predicted_discount[:, :-1]
    
    td_1 = cur_rewards + predicted_discount * next_values * (1 - lambda_)
    returns = torch.zeros_like(cur_rewards).to(device)
    
    
    for i in reversed(range(td_1.size(1))):
        last_lambda = td_1[:, i] + predicted_discount[:, i] * lambda_ * last_lambda
        returns[:, i] = last_lambda
        
    return returns