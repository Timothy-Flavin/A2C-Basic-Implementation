import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt


# Value Network for deep RL
class V(nn.Module):
    def __init__(self, input_size, hidden_dims=[64, 64], output_size=1, device="cuda"):
        super(V, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers = nn.ModuleList(self.layers)
        self.to(device)

    def forward(self, x):
        for i, la in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = torch.relu(la(x))
            else:
                x = la(x)
        return x


class Actor_Critic:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_sizes=[128, 128],
        gamma=0.99,
        V_learning_rate=0.001,
        A_learning_rate=0.0003,
        batch_size=128,
        device="cuda",
        entropy_coeficient=0.001,
    ):
        # TODO
        # Learn Value Function V
        self.e_coef = entropy_coeficient
        self.device = device
        print(device)
        self.V = V(
            state_size, hidden_dims=hidden_sizes, output_size=action_size, device=device
        )
        self.V_optimizer = optim.AdamW(
            self.V.parameters(), lr=V_learning_rate, amsgrad=True
        )

        self.pi = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sizes[1], action_size),
        ).to(self.device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=A_learning_rate)

        self.gamma = gamma
        self.action_size = action_size

    def select_action(self, obs, env):
        with torch.no_grad():
            act = self.pi(torch.FloatTensor(obs).to(self.device))
            probs = torch.nn.Softmax(dim=-1)(act)
            return np.random.choice(self.action_size, p=probs.cpu().numpy())

    def v_loss(self, S, A, S_, R, dones):
        # loss = ( reward + (1-done)*gamma*(V(s')) - V(s) )^2
        # We want to use the temporal difference loss here
        v = self.V(S)  # Getting values for states
        v_ = None
        with torch.no_grad():  # No reason to track gradient for the target Q network
            v_ = R + (1 - dones) * self.gamma * self.V(S_)
        adv = v_ - v
        v_loss = (adv) ** 2
        return v_loss, adv

    def update(self, state, action, reward, state_, terminated, truncated):
        A = (
            torch.Tensor(np.array([action])).long().to(self.device)
        )  # Moving actions to a tensor
        S = torch.FloatTensor(state)[None, :].to(self.device)
        S_ = torch.FloatTensor(state_)[None, :].to(self.device)
        R = torch.FloatTensor(np.array([reward])).to(self.device)
        D = torch.Tensor(np.array([int(terminated)])).long().to(self.device)

        # Critic update
        loss, adv = self.v_loss(S, A, S_, R, D)
        # print(loss)
        self.V_optimizer.zero_grad()
        loss.backward()
        self.V_optimizer.step()

        # Actor Update
        prob = torch.nn.Softmax(dim=-1)(self.pi(S))  # Get the probabilities from pi
        log_probs = torch.log(prob)  # Get the log probabilities
        entropy = -(prob * log_probs).sum(dim=1)  # Calculate entropy
        log_probs = log_probs[
            :, A.item()
        ]  # .gather(1, A.view(-1, 1)).view(-1) #Get the log probs for the actions we actually took

        self.pi_optim.zero_grad()
        objective = -(
            adv.detach() * log_probs - self.e_coef * entropy
        )  # Loss is policy gradient loss - entropy because we want the policy to not converge too fast
        # print(f"adv: {adv.detach() * log_probs}, ent: {self.e_coef*entropy}")
        objective.backward()
        self.pi_optim.step()

    def save(self, path):
        torch.save(self.V, path + "_V")
        torch.save(self.pi, path + "_pi")

    def load(self, path):
        self.pi.load_state_dict(torch.load(path + "_pi").state_dict())
        self.V.load_state_dict(torch.load(path + "_V").state_dict())


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    n_episodes = 500

    env = gym.make("CartPole-v1")
    agent = Actor_Critic(
        state_size=4,
        action_size=2,
        hidden_sizes=[256, 256],
        gamma=0.99,
        V_learning_rate=1e-3,
        A_learning_rate=3e-5,
        device="cuda",
    )

    rewards_over_time = []
    for i in range(n_episodes):
        tot_reward = 0
        obs, info = env.reset()
        tot_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.select_action(obs, env)  # int(input("Action: "))#
            # print(action)
            obs_, reward, terminated, truncated, info = env.step(
                action
            )  # = int(input("Action: "))
            agent.update(
                obs, action, reward, obs_, terminated, truncated
            )  # Agent update
            obs = obs_
            tot_reward += reward
        print(f"Reward for ep {i}:{tot_reward}")
        rewards_over_time.append(tot_reward)

    plt.plot(rewards_over_time)
    plt.title("Reward over time")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid()
    plt.show()
