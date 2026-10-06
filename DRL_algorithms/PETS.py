import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_utils import ReplayBuffer


def make_env(env_id: str, normalize: bool):
    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def parse_args():
    parser = argparse.ArgumentParser(description="PETS")
    parser.add_argument("--exp_name", type=str, default="PETS")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    # Pendulum-v1, LunarLanderContinuous-v2, BipedalWalker-v3, Walker2d-v4, HalfCheetah-v4, Ant-v4, Swimmer-v4, Hopper-v4
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=5000)
    parser.add_argument("--buffer_size", type=int, default=5000)
    # dynamics ensemble model
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--num_elite_models", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--validation_ratio", type=float, default=0.0)
    parser.add_argument("--train_epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning_starts", type=int, default=200)
    parser.add_argument("--train_freq", type=int, default=50)
    # planning / cem
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--cem_iters", type=int, default=5)
    parser.add_argument("--cem_alpha", type=float, default=0.1)
    parser.add_argument("--population_size", type=int, default=350)
    parser.add_argument("--num_particles", type=int, default=20)
    parser.add_argument("--elite_ratio", type=float, default=0.1)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


class EnsembleLinearLayer(nn.Module):
    """Weights are initialized from a truncated normal distribution and biases are set to zero"""
    def __init__(self, num_models: int, in_features: int, out_features: int):
        super().__init__()
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_models, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(num_models, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (2 * self.in_features ** 0.5)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.bias)

    def forward(self, x, model_idx=None):
        weight, bias = self.weight, self.bias
        if model_idx is not None:
            weight, bias = weight[model_idx], bias[model_idx]
        return torch.bmm(x, weight) + bias


class EnsembleNetwork(nn.Module):
    """probabilistic dynamics ensemble, predicts delta observation and reward"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int,
                 num_models: int, lr: float):
        super().__init__()
        self.num_models = num_models
        out_dim = obs_dim + 1
        dims = [obs_dim + action_dim] + [hidden_dim] * num_layers
        self.hidden_layers = nn.ModuleList([
            EnsembleLinearLayer(num_models, dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.mean_layer = EnsembleLinearLayer(num_models, hidden_dim, out_dim)
        self.log_var_layer = EnsembleLinearLayer(num_models, hidden_dim, out_dim)
        self.log_var_min = nn.Parameter(torch.full((out_dim,), -10.0), requires_grad=False)
        self.log_var_max = nn.Parameter(torch.full((out_dim,), 2.0), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, a, model_idx=None):
        h = torch.cat([x, a], dim=-1)
        for layer in self.hidden_layers:
            h = F.silu(layer(h, model_idx))
        mean = self.mean_layer(h, model_idx)
        log_var = self.log_var_layer(h, model_idx)
        log_var = self.log_var_max - F.softplus(self.log_var_max - log_var)
        log_var = self.log_var_min + F.softplus(log_var - self.log_var_min)
        return mean, log_var

    def nll_loss(self, s, a, target):
        """Gaussian NLL Loss"""
        mean, log_var = self.forward(s, a)
        var = log_var.exp() + 1e-6
        nll = (target - mean).pow(2) / var + log_var
        return nll.mean(dim=(1, 2)).sum() + 0.01 * (self.log_var_max.sum() - self.log_var_min.sum())

    def update(self, s, a, target):
        loss = self.nll_loss(s, a, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def mse_loss(self, s, a, target):
        """used for validation and elite selection"""
        mean, _ = self.forward(s, a)
        return (target - mean).pow(2).mean(dim=(1, 2))


class FakeEnv:
    """environment for trajectory sampling"""
    def __init__(self, model: EnsembleNetwork, obs_dim: int, num_particles: int):
        self.model = model
        self.obs_dim = obs_dim
        self.num_particles = num_particles
        self.device = next(model.parameters()).device
        self.elite_models = torch.arange(model.num_models, device=self.device)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, act: torch.Tensor):
        """obs (G, B, obs_dim), act (G, B, action_dim) -> rewards, next_obs (G, B, ...)"""
        num_members = self.elite_models.shape[0]
        model_idx = self.elite_models[torch.randint(num_members, (obs.shape[0],), device=self.device)]
        mean, log_var = self.model(obs, act, model_idx)
        std = log_var.mul(0.5).exp()
        sample = mean + std * torch.randn_like(mean)
        next_obs = obs + sample[..., :self.obs_dim]
        rewards = sample[..., self.obs_dim]
        return rewards, next_obs

    @torch.no_grad()
    def rollout(self, obs: torch.Tensor, actions: torch.Tensor):
        """trajectory sampling TS1"""
        population_size, horizon, _ = actions.shape
        group = population_size * self.num_particles
        states = obs.view(1, 1, -1).expand(group, 1, self.obs_dim)
        returns = torch.zeros(group, 1, device=self.device)
        for t in range(horizon):
            a_t = actions[:, t].repeat_interleave(self.num_particles, dim=0).unsqueeze(1)
            rewards, states = self.step(states, a_t)
            returns += rewards
        return returns.view(population_size, self.num_particles).mean(dim=1)


class PETSAgent:
    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = args.horizon
        self.cem_iters = args.cem_iters
        self.cem_alpha = args.cem_alpha
        self.population_size = args.population_size
        self.num_elite_models = args.num_elite_models
        self.elite_ratio = args.elite_ratio
        self.validation_ratio = args.validation_ratio
        self.train_epochs = args.train_epochs
        self.patience = args.patience
        self.batch_size = args.batch_size

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.action_center = (self.action_low + self.action_high) / 2.0

        self.model = EnsembleNetwork(
            obs_dim, action_dim, args.hidden_dim, args.num_layers, args.num_models, args.lr,
        ).to(self.device)
        self.prev_solution = None  # last MPC plan, used to warm-start the CEM mean
        self.model_env = FakeEnv(self.model, obs_dim, args.num_particles)

    def model_train(self, buffer):
        s, a, r, s_, _ = [t.float().to(self.device) for t in buffer.sample_all()]
        target = torch.cat([s_ - s, r.unsqueeze(-1)], dim=-1)

        num_samples = s.shape[0]
        val_size = int(self.validation_ratio * num_samples)
        train_end = num_samples - val_size
        val = slice(train_end, num_samples)
        if val_size == 0:
            val = slice(0, train_end)

        # validation uses the same data for every member
        s_val = s.unsqueeze(0).expand(self.model.num_models, -1, -1)
        a_val = a.unsqueeze(0).expand(self.model.num_models, -1, -1)
        target_val = target.unsqueeze(0).expand(self.model.num_models, -1, -1)

        best_val_loss = float("inf")
        best_state = None
        train_loss = 0.0
        epochs_no_improve = 0
        epoch = 0
        # train_epochs=0 means stopping is decided by patience alone
        while self.train_epochs == 0 or epoch < self.train_epochs:
            epoch += 1
            # shuffle the training data independently for every ensemble member
            perm = torch.stack([torch.randperm(train_end, device=self.device) for _ in range(self.model.num_models)])
            total_loss = 0.0
            steps = 0
            for start in range(0, train_end, self.batch_size):
                end = min(start + self.batch_size, train_end)
                idx = perm[:, start:end]
                total_loss += self.model.update(s[idx], a[idx], target[idx])
                steps += 1
            train_loss = total_loss / steps
            val_loss = self.model.mse_loss(s_val[:, val], a_val[:, val], target_val[:, val]).sum().item()
            improvement = float("inf") if best_state is None else (best_val_loss - val_loss) / best_val_loss
            if improvement > 0.01:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break
        self.model.load_state_dict(best_state)  # keep the best model selected by validation

        # pick the members for planning
        val_losses = self.model.mse_loss(s_val[:, val], a_val[:, val], target_val[:, val])
        self.model_env.elite_models = torch.topk(val_losses, self.num_elite_models, largest=False).indices
        return train_loss, best_val_loss

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        """model predictive control with cem"""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        num_cem_elites = max(1, int(self.elite_ratio * self.population_size))

        if self.prev_solution is None:
            mean = self.action_center.repeat(self.horizon, 1)
        else:
            mean = torch.cat([self.prev_solution[1:], self.prev_solution[-1:]], dim=0)
        std = (self.action_high - self.action_low).repeat(self.horizon, 1) / 4.0

        for _ in range(self.cem_iters):
            # mean and std are (horizon, action_dim) and broadcast over the population
            noise = torch.randn(self.population_size, self.horizon, self.action_dim, device=self.device)
            actions = mean + std * noise
            actions = torch.where(actions > self.action_low, actions, self.action_low)
            actions = torch.where(actions < self.action_high, actions, self.action_high)
            returns = self.model_env.rollout(obs_t, actions)
            elites = actions[torch.topk(returns, num_cem_elites).indices]
            mean = self.cem_alpha * mean + (1.0 - self.cem_alpha) * elites.mean(dim=0)
            std = self.cem_alpha * std + (1.0 - self.cem_alpha) * elites.std(dim=0, correction=0)
        self.prev_solution = mean
        return mean[0].cpu().numpy()


def train(args):
    env = make_env(args.env, args.normalize)()
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(env.action_space, gym.spaces.Box), "PETS only supports continuous action spaces"

    obs_shape = env.observation_space.shape
    obs_dim = int(np.array(obs_shape).prod())
    action_dim = int(env.action_space.shape[0])
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    buffer = ReplayBuffer(args.buffer_size)
    agent = PETSAgent(obs_dim, action_dim, low, high, args)

    writer = SummaryWriter(f"runs/{run_name}")
    # log hyperparameters
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    start_time = time.time() - 1e-6
    last_log_step = 0

    while global_step < args.total_timesteps:
        if global_step >= args.learning_starts and global_step % args.train_freq == 0:
            train_loss, val_loss = agent.model_train(buffer)
            writer.add_scalar("loss/train_nll", train_loss, global_step)
            writer.add_scalar("loss/val_mse", val_loss, global_step)

        # model predictive control
        if global_step < args.learning_starts:
            action = env.action_space.sample()  # random policy for initial exploration
        else:
            action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add((obs, action, reward, next_obs, terminated))
        obs = next_obs
        global_step += 1

        # log episode return and episode length
        if "episode" in info:
            episode_reward = float(info["episode"]["r"])
            episode_length = int(info["episode"]["l"])
            writer.add_scalar("charts/return", episode_reward, global_step)
            writer.add_scalar("charts/length", episode_length, global_step)
            print(f"global_step={global_step}, episodic_return={episode_reward:.3f}")

            obs, _ = env.reset()

        # log steps per second
        if global_step - last_log_step >= 50:
            sps = global_step / (time.time() - start_time)
            writer.add_scalar("charts/sps", sps, global_step)
            print(f"SPS={int(sps)}")
            last_log_step = global_step

    writer.close()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
