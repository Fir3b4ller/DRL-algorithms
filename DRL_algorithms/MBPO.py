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


def make_env(env_id: str, gamma: float, normalize: bool):
    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def parse_args():
    parser = argparse.ArgumentParser(description="MBPO")
    parser.add_argument("--exp_name", type=str, default="MBPO")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    # Pendulum-v1, LunarLanderContinuous-v2, BipedalWalker-v3, Walker2d-v4, HalfCheetah-v4, Ant-v4, Swimmer-v4, Hopper-v4
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=5000)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--epoch_length", type=int, default=200)
    parser.add_argument("--learning_starts", type=int, default=500)
    # soft actor-critic
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto_tune_alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_sac_updates", type=int, default=20)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--target_network_frequency", type=int, default=4)
    parser.add_argument("--real_ratio", type=float, default=0.05)
    # dynamics ensemble model
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--num_elite_models", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--model_lr", type=float, default=1e-3)
    parser.add_argument("--model_batch_size", type=int, default=256)
    parser.add_argument("--model_train_freq", type=int, default=200)
    parser.add_argument("--model_train_epochs", type=int, default=0)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    # model rollouts
    parser.add_argument("--model_rollouts_per_environment_step", type=int, default=400)
    parser.add_argument("--model_buffer_size", type=int, default=80000)
    parser.add_argument("--rollout_min_length", type=int, default=1)
    parser.add_argument("--rollout_max_length", type=int, default=1)
    parser.add_argument("--rollout_schedule_start", type=int, default=1)
    parser.add_argument("--rollout_schedule_end", type=int, default=15)
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
    """learned environment used to generate model rollouts"""
    def __init__(self, model: EnsembleNetwork, obs_dim: int):
        self.model = model
        self.obs_dim = obs_dim
        self.device = next(model.parameters()).device
        self.elite_models = torch.arange(model.num_models, device=self.device)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, act: torch.Tensor):
        """obs (B, obs_dim), act (B, action_dim) -> rewards (B,), next_obs (B, obs_dim)"""
        num_members = self.elite_models.shape[0]
        model_idx = self.elite_models[torch.randint(num_members, (obs.shape[0],), device=self.device)]
        mean, log_var = self.model(obs.unsqueeze(1), act.unsqueeze(1), model_idx)
        std = log_var.mul(0.5).exp()
        sample = (mean + std * torch.randn_like(mean)).squeeze(1)
        next_obs = obs + sample[..., :self.obs_dim]
        rewards = sample[..., self.obs_dim]
        return rewards, next_obs


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return self.mean(h), log_std

    def get_action(self, x: torch.Tensor):
        """reparameterized sample"""
        mean, log_std = self.forward(x)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        action = torch.tanh(u)
        # Jacobian correction
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-7).sum(-1)
        return action, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, a], dim=-1)).squeeze(-1)


class MBPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.real_ratio = args.real_ratio
        self.auto_tune_alpha = args.auto_tune_alpha
        self.target_entropy = -action_dim
        self.epoch_length = args.epoch_length
        self.rollout_min_length = args.rollout_min_length
        self.rollout_max_length = args.rollout_max_length
        self.rollout_schedule_start = args.rollout_schedule_start
        self.rollout_schedule_end = args.rollout_schedule_end
        self.rollout_batch_size = args.model_rollouts_per_environment_step * args.model_train_freq
        self.num_elite_models = args.num_elite_models
        self.validation_ratio = args.validation_ratio
        self.model_train_epochs = args.model_train_epochs
        self.patience = args.patience
        self.model_batch_size = args.model_batch_size

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.action_center = (self.action_low + self.action_high) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0

        # probabilistic dynamics ensemble
        self.model = EnsembleNetwork(
            obs_dim, action_dim, args.hidden_dim, args.num_layers, args.num_models, args.model_lr,
        ).to(self.device)
        self.fake_env = FakeEnv(self.model, obs_dim)

        # soft actor-critic
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=args.q_lr)
        # entropy coefficient
        if self.auto_tune_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(np.log(args.alpha), dtype=torch.float32, device=self.device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.log_alpha = torch.tensor(np.log(args.alpha), dtype=torch.float32, device=self.device)

    def get_rollout_length(self, global_step: int):
        epoch = global_step // self.epoch_length
        if epoch <= self.rollout_schedule_start:
            return self.rollout_min_length
        dx = min((epoch - self.rollout_schedule_start) / (self.rollout_schedule_end - self.rollout_schedule_start), 1.0)
        return int(dx * (self.rollout_max_length - self.rollout_min_length) + self.rollout_min_length)

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
        # model_train_epochs=0 means stopping is decided by patience alone
        while self.model_train_epochs == 0 or epoch < self.model_train_epochs:
            epoch += 1
            # shuffle the training data independently for every ensemble member
            perm = torch.stack([torch.randperm(train_end, device=self.device) for _ in range(self.model.num_models)])
            total_loss = 0.0
            steps = 0
            for start in range(0, train_end, self.model_batch_size):
                end = min(start + self.model_batch_size, train_end)
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

        # pick the members used for model rollouts
        val_losses = self.model.mse_loss(s_val[:, val], a_val[:, val], target_val[:, val])
        self.fake_env.elite_models = torch.topk(val_losses, self.num_elite_models, largest=False).indices
        return train_loss, best_val_loss

    @torch.no_grad()
    def model_rollout(self, real_buffer, model_buffer, rollout_length: int):
        """rollouts with the current policy, added to the model buffer"""
        num_rollouts = self.rollout_batch_size
        obs = real_buffer.sample(num_rollouts)[0].float().to(self.device)
        for _ in range(rollout_length):
            act = self.select_action(obs)
            rewards, next_obs = self.fake_env.step(obs, act)
            model_buffer.add_batch((
                obs.cpu().numpy(),
                act.cpu().numpy(),
                rewards.cpu().numpy(),
                next_obs.cpu().numpy(),
                np.zeros(num_rollouts, dtype=bool),
            ))
            obs = next_obs
        return num_rollouts

    @torch.no_grad()
    def select_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        u = self.actor.get_action(obs)[0]
        return self.action_center + u * self.action_scale

    def SAC_update(self, real_buffer, model_buffer, update_actor: bool, update_target: bool):
        # mix real data and model data for training
        num_real = max(1, int(self.real_ratio * self.batch_size))
        num_model = self.batch_size - num_real
        if len(model_buffer) >= num_model:
            real_batch = real_buffer.sample(num_real)
            model_batch = model_buffer.sample(num_model)
            batch = [torch.cat([rb, mb], dim=0) for rb, mb in zip(real_batch, model_batch)]
        else:
            batch = real_buffer.sample(self.batch_size)
        s, a, r, s_, done = [t.to(self.device) for t in batch]
        alpha = self.log_alpha.exp()

        # critic
        with torch.no_grad():
            next_a, next_log_prob = self.actor.get_action(s_)
            next_a_scaled = self.action_center + next_a * self.action_scale
            next_q = torch.min(self.target_critic1(s_, next_a_scaled), self.target_critic2(s_, next_a_scaled))
            target_q = r + self.gamma * (1.0 - done) * (next_q - alpha * next_log_prob)

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor + alpha
        actor_loss = alpha_loss = None
        if update_actor:
            pi_a, pi_log_prob = self.actor.get_action(s)
            pi_a_scaled = self.action_center + pi_a * self.action_scale
            actor_loss = (alpha.detach() * pi_log_prob - torch.min(self.critic1(s, pi_a_scaled), self.critic2(s, pi_a_scaled))).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.auto_tune_alpha:
                alpha_loss = (-alpha * (pi_log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        # soft update
        if update_target:
            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        return critic_loss.item(), actor_loss, float(torch.min(q1, q2).mean().item()), alpha_loss


def train(args: argparse.Namespace):
    env = make_env(args.env, args.gamma, args.normalize)()
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(env.action_space, gym.spaces.Box), "MBPO only supports continuous action spaces"

    obs_shape = env.observation_space.shape
    action_space = env.action_space
    obs_dim = int(np.array(obs_shape).prod())
    action_dim = int(action_space.shape[0])
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    real_buffer = ReplayBuffer(args.buffer_size)
    model_buffer = ReplayBuffer(args.model_buffer_size)
    agent = MBPOAgent(obs_dim, action_dim, low, high, args)

    writer = SummaryWriter(f"runs/{run_name}")
    # log hyperparameters
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    sac_step = 0
    start_time = time.time()
    last_log_step = 0

    while global_step < args.total_timesteps:
        # collect real data
        if global_step < args.learning_starts:
            action = action_space.sample()  # random policy for initial exploration
        else:
            action = agent.select_action(obs).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        real_buffer.add((obs, action, reward, next_obs, terminated))
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

        # train the dynamics model and generate model rollouts
        if global_step >= args.learning_starts and global_step % args.model_train_freq == 0:
            train_loss, val_loss = agent.model_train(real_buffer)
            rollout_length = agent.get_rollout_length(global_step)
            agent.model_rollout(real_buffer, model_buffer, rollout_length)
            writer.add_scalar("model/train_nll", train_loss, global_step)
            writer.add_scalar("model/val_mse", val_loss, global_step)
            writer.add_scalar("model/rollout_length", rollout_length, global_step)

        # optimize the policy on real + model data
        if global_step >= args.learning_starts:
            for _ in range(args.num_sac_updates):
                update_actor = sac_step % args.policy_frequency == 0
                update_target = sac_step % args.target_network_frequency == 0
                critic_loss, actor_loss, mean_q, alpha_loss = agent.SAC_update(
                    real_buffer, model_buffer, update_actor, update_target
                )
                sac_step += 1
            if global_step % 100 == 0:
                writer.add_scalar("loss/critic_loss", critic_loss, global_step)
                writer.add_scalar("loss/q_value", mean_q, global_step)
                if actor_loss is not None:
                    writer.add_scalar("loss/actor_loss", actor_loss.item(), global_step)
                if alpha_loss is not None:
                    writer.add_scalar("loss/alpha_loss", alpha_loss.item(), global_step)
                    writer.add_scalar("charts/alpha", agent.log_alpha.exp().item(), global_step)


        # log steps per second
        if global_step - last_log_step >= 100:
            sps = global_step / (time.time() - start_time)
            writer.add_scalar("charts/sps", sps, global_step)
            print(f"SPS={int(sps)}")
            last_log_step = global_step

    writer.close()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
