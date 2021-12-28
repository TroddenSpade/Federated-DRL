import numpy as np
from tqdm import tqdm

from pytorch.Agent import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer

class Federator:
    def __init__(self, n_agents, update_rate, args) -> None:
        
        self.env = args["env_fn"]()
        self.n_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape

        self.global_agent = Agent(**args)

        self.update_rate = update_rate
        self.n_agents = n_agents
        self.agents = []
        for _ in range(n_agents):
            agent = Agent(**args)
            self.agents.append(agent)

        self.set_local_networks()

            
    def train(self, n_runs):
        rewards = np.zeros(n_runs)
        for r in tqdm(range(n_runs)):
            for agent in self.agents:
                agent.step(self.update_rate)
            self.aggregate_networks()
            self.set_local_networks()
            rewards[r] = self.global_agent.evaluate()
        print(self.agents[0].episode_count)
        return rewards


    def aggregate_networks(self):
        sd_online = self.global_agent.online_net.state_dict()
        sd_target = self.global_agent.target_net.state_dict()

        online_dicts = []
        target_dicts = []
        for agent in self.agents:
            online_dicts.append(agent.online_net.state_dict())
            target_dicts.append(agent.target_net.state_dict())

        for key in sd_online:
            sd_online[key] -= sd_online[key]
            for dict in online_dicts:
                sd_online[key] += dict[key]
            sd_online[key] /= self.n_agents

        for key in sd_target:
            sd_target[key] -= sd_target[key]
            for dict in target_dicts:
                sd_target[key] += dict[key]
            sd_target[key] /= self.n_agents


    def set_local_networks(self):
        for agent in self.agents:
            agent.online_net.load_state_dict(
                self.global_agent.online_net.state_dict())
            agent.target_net.load_state_dict(
                self.global_agent.target_net.state_dict())