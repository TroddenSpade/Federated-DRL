import numpy as np

class ReplayBuffer:
    def __init__(self, state_shape, action_space, 
                 batch_size=64, max_size=50000):
        self.next = 0
        self.size = 0
        self.max_size = max_size
        self.batch_size = batch_size

        self.states = np.empty(shape=(max_size, *state_shape))
        self.actions = np.empty(shape=(max_size, action_space))
        self.rewards = np.empty(shape=(max_size))
        self.states_p = np.empty(shape=(max_size, *state_shape))
        self.is_terminals = np.empty(shape=(max_size), dtype=np.float)

    def __len__(self): return self.size
    
    def store(self, state, action, reward, state_p, is_terminal):
        self.states[self.next] = state
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.states_p[self.next] = state_p
        self.is_terminals[self.next] = is_terminal

        self.next += 1
        self.size = min(self.size + 1, self.max_size)
        self.next = self.next % self.max_size

    def sample(self, batch_size=None):
        batch_size = self.batch_size \
                        if batch_size is None else batch_size
        indices = np.random.choice(self.size, size=batch_size,
                                   replace=False)
        return self.states[indices], \
            self.actions[indices], \
            self.rewards[indices], \
            self.states_p[indices], \
            self.is_terminals[indices]

    def clear(self):
        self.next = 0
        self.size = 0
        self.states = np.empty_like(self.states)
        self.actions = np.empty_like(self.actions)
        self.rewards = np.empty_like(self.rewards)
        self.states_p = np.empty_like(self.states_p)
        self.is_terminals = np.empty_like(self.is_terminals)
