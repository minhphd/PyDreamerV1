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


def get_obs(env, new_size):
    obs = env.physics.render(*new_size)
    obs = obs/255 - 0.5
    rearranged_obs = obs.transpose([2,0,1])
    return rearranged_obs


def td_lambda(rewards, dones, values, lamda_val, discount_val, device):
    """
    Compute the TD(λ) returns for value estimation.

    Args:
    - states (Tensor): Tensor of states with shape [batch_size, time_steps, state_dim].
    - rewards (Tensor): Tensor of rewards with shape [batch_size, time_steps].
    - dones (Tensor): Tensor indicating episode termination with shape [batch_size, time_steps].
    - values (Tensor): Tensor of value estimates with shape [batch_size, time_steps].
    - lamda_val (float): The λ parameter in TD(λ) controlling bias-variance tradeoff.
    - discount_val (float): Discount factor (γ) used in calculating returns.

    Returns:
    - td_lambda (Tensor): The computed lambda returns with shape [batch_size, time_steps - 1].
    """
    # Exclude the last timestep for rewards and dones
    rewards = rewards[:, :-1]
    dones = dones[:, :-1]

    # Initialize td_lambda tensor with one less timestep
    td_lambda = torch.zeros_like(rewards).to(device)

    # Use next state's value if not done, else use 0 (as the episode ends)
    next_values = values[:, 1:] * (1 - dones)
    
    # Bootstrap from next state's value
    future_return = next_values[:, -1]

    for t in reversed(range(rewards.size(1))):
        td_target = rewards[:, t] + discount_val * future_return
        future_return = values[:, t] + lamda_val * (td_target - values[:, t])
        td_lambda[:, t] = future_return

    return td_lambda


def td_lambda_exp(rewards, predicted_discount, values, lambda_val, device):
    """
    Compute the TD(λ) returns for value estimation.

    Args:
    - states (Tensor): Tensor of states with shape [batch_size, time_steps, state_dim].
    - rewards (Tensor): Tensor of rewards with shape [batch_size, time_steps].
    - predicted_discount (Tensor): Tensor indicating probability of episode termination with shape [batch_size, time_steps].
    - values (Tensor): Tensor of value estimates with shape [batch_size, time_steps].
    - lamda_val (float): The λ parameter in TD(λ) controlling bias-variance tradeoff.

    Returns:
    - td_lambda (Tensor): The computed lambda returns with shape [batch_size, time_steps - 1].
    """
    last_lambda = 0
    lambda_targets = torch.zeros_like((rewards[:, :-1])).to(device)
    
    HORIZON = rewards.size(1)
    for step in reversed(range(HORIZON-1)):
        delta = rewards[:, step] + values[:, step + 1] * predicted_discount[:, step]
        last_lambda = delta + lambda_val * predicted_discount[:, step] * last_lambda
        lambda_targets[:, step] = last_lambda
        
    return lambda_targets