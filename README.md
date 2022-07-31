# Parallel Deep Reinforcement Learning with Federated Learning Framework
The purpose of this project is to assess the effect of parallel training of multiple Deep Reinforcement Learning agents using the Federated Averaging (FedAVG) algorithm -- in other words, after training the agents for a specific timesteps, all of the Deep Q Network models are aggregated by taking average of their parameters and subsequently the averaged model will be set for all of the agents for more training rounds.

### Environments
* CartPole
* Lunar Lander
* Super Mario Bros

### Deep Reinforcement Learning Methods
* Deep Q Network
* Double Deep Q Network

## Experiments
### 3 DQN Agents on Cartpole Environemt
![CP1](https://github.com/TroddenSpade/Federated-DQN/blob/main/results/CartPole/1/CartPole.png?raw=true)
![CP2](https://github.com/TroddenSpade/Federated-DQN/blob/main/results/CartPole/2/Figure_2.png?raw=true)

### 3 DQN Agents on Lunar Lander Environemt
![LL](https://github.com/TroddenSpade/Federated-DQN/blob/main/results/LunarLander/lunarlander.png?raw=true)

### 4 DDQN Agents on Super Mario Bros 1-1 to 1-4
![SMB](https://github.com/TroddenSpade/Federated-DQN/blob/main/results/Mario/rewards.png?raw=true)

![1-2](https://github.com/TroddenSpade/Federated-DQN/blob/main/results/Mario/1.gif?raw=true)
![1-4](https://github.com/TroddenSpade/Federated-DQN/blob/main/results/Mario/3.gif?raw=true)
