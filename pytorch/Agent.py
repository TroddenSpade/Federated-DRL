import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer

class Agent():
    def __init__(self, id, env_name, env_fn, Qnet=QNetwork, buffer=ReplayBuffer,
                 max_epsilon=1, min_epsilon=0.05, epsilon_decay=0.99, gamma=0.9,
                 target_update_rate=2000, min_buffer=100, 
                 load=False, path=None) -> None:
        self.id = id
        self.path = path + str(id) + "/"

        self.env = env_fn(env_name)
        self.env_fn = env_fn
        self.n_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.min_buffer = min_buffer
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_rate = target_update_rate
        self.buffer = buffer(self.state_shape, self.n_actions,
                             load=load, path=self.path)

        self.online_net = Qnet(self.state_shape, self.n_actions).to(self.device)
        self.target_net = Qnet(self.state_shape, self.n_actions).to(self.device)

        if load:
            self.load()
        else:
            self.update_target_network()
            self.epsilon = max_epsilon
            self.step_count = 0
            self.episode_count = 0
            self.rewards = []

    
    def load(self):
        with open(self.path + "step_count.pkl", 'rb') as f:
            self.step_count = pickle.load(f)
        with open(self.path + "episode_count.pkl", 'rb') as f:
            self.episode_count = pickle.load(f)
        with open(self.path + "rewards.pkl", 'rb') as f:
            self.rewards = pickle.load(f)
        with open(self.path + "epsilon.pkl", 'rb') as f:
            self.epsilon = pickle.load(f)
        self.online_net.load_state_dict(torch.load(self.path + "online_net.pt", 
                                                   map_location=torch.device(self.device)))
        self.target_net.load_state_dict(torch.load(self.path + "target_net.pt", 
                                                   map_location=torch.device(self.device)))

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.buffer.save()
        with open(self.path + "step_count.pkl", "wb") as f:
            pickle.dump(self.step_count, f)
        with open(self.path + "episode_count.pkl", "wb") as f:
            pickle.dump(self.episode_count, f)
        with open(self.path + "rewards.pkl", "wb") as f:
            pickle.dump(self.rewards, f)
        with open(self.path + "epsilon.pkl", "wb") as f:
            pickle.dump(self.epsilon, f)
        torch.save(self.online_net.state_dict(), self.path +  "online_net.pt")
        torch.save(self.target_net.state_dict(), self.path +  "target_net.pt")



    def train(self, n_episodes):
        for i in tqdm(range(n_episodes)):
            episode_reward = 0
            state = self.env.reset()

            while True:
                self.step_count += 1
                action = self.epsilonGreedyPolicy(state)
                state_p, reward, done, info = self.env.step(action)
                episode_reward += reward

                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_failure = done and not is_truncated
                self.buffer.store(state, action, reward, state_p, float(is_failure))

                if len(self.buffer) >= self.min_buffer:
                    self.update()
                    if self.step_count % self.target_update_rate == 0:
                        self.update_target_network()

                state = state_p
                if done:
                    self.episode_count += 1
                    self.rewards.append(episode_reward)
                    break

        print("Agent-{} Episode {} Step {} score = {}, average score = {}"\
                .format(self.id, self.episode_count, self.step_count, self.rewards[-1], np.mean(self.rewards)))


    def get_score(self):
        # return np.mean(self.rewards[-5:])
        return 1


    def update(self):
        states, actions, rewards, states_p, is_terminals = self.buffer.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_p = states_p.to(self.device)
        is_terminals = is_terminals.to(self.device)

        td_estimate = self.online_net(states).gather(1, actions)

        actions_p = self.online_net(states).argmax(axis=1, keepdim=True)
        with torch.no_grad():
            q_states_p = self.target_net(states_p)
        q_state_p_action_p = q_states_p.gather(1, actions_p)
        td_target = rewards + (1-is_terminals) * self.gamma * q_state_p_action_p

        self.online_net.update_netowrk(td_estimate, td_target)
        self.update_epsilon()


    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)


    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def epsilonGreedyPolicy(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = state.__array__()
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.online_net(state).argmax().item()
        return action
