import numpy as np

class Agent():
    def __init__(self, env_fn, Qnet, buffer, 
                 epsilon, target_update_rate) -> None:
        
        # self.online_net = Qnet()
        # self.target_net = Qnet()
        # self.update_target_network()

        self.env = env_fn()
        print(self.env.action_space.n)
        print(self.env.observation_space.shape)

        self.buffer = buffer()
        self.epsilon = epsilon
        self.target_update_rate = target_update_rate


    def step(self):
        pass
    

    def train(self, n_episodes):
        step_count = 0
        rewards = np.zeros(n_episodes)

        for i in range(n_episodes):
            state = self.env.reset()
            while True:
                step_count += 1
                action = self.epsilonGreedyPolicy(state)
                state_p, reward, done, info = self.env.step()
                rewards[i] += reward

                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_failure = done and not is_truncated
                self.buffer.store((state, action, reward, state_p, float(is_failure)))

                if len(self.replay_buffer) >= self.min_buffer:
                    batches = self.replay_buffer.sample()
                    self.update(batches)

                if step_count % self.target_update_rate == 0:
                    self.update_target_network()

                if done:
                    break

                state = state_p
        return rewards


    def update(self, batches):
        pass


    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def epsilonGreedyPolicy(self):
        if np.random.rand() < self.epsilon:
            pass
            # of actions ?

    
    def greedyPolicy(self):
        pass

    