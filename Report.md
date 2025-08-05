## Multi Agent Tennis playing using DDPG
Justin Hess

## DDPG Algorithm

Deep Deterministic Policy Gradient (DDPG) is an Actor-Critic method with a twist. The actor produces a deterministic policy by mapping states to actions instead of a standard stochastic policy, and the critic evaluates actor's policy output. The critic is updated using the temporal difference (TD) error and the actor is trained using the deterministic policy gradient algorithm. The critic neural network in DDPG is used to approximate the maximizer over the Q-values of the next state, and not as a learned baseline, as have seen so far. One of the limitations of the DQN agent is that it is not straightforward to use in continuous action spaces. For example, how do you get the value of continuous action with DQN architecture? This is the problem DDPG solves.

The reinforcement learning agent implementation follows the ideas of [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) paper implementing a DDPG agent.

The actor network is responsible for chosing actions based on the state and the critic network try to estimate the reward for the given state-action pair. The actor network is indirectly trained using gradient ascent on the critic network, reducing the problem of building a loss function to a more classic RL problem of maximize the expected reward.

The agent exploits the initial lack of knowledge using additive random noise to explore the environment [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process).

The hyperparameters used are:

- Actor learning rate: 0.0001
- Critic learning rate: 0.0001
- Batch size: 128
- Gamma: 0.99
- Tau: 0.001
- Adam weight decay: 0

The number of episodes used is 5,000, which is not a hyperparameter.

## Multi Agent Learning using DDPG

The agents learn by using the actor network to choose continuous actions from evaluating states, and then inputting those actions to the critic network which outputs a singular Q value (like in DQN learning's target network) for the action from the actor.

Q_target = r + γ * Q_target(s', μ_target(s'))

μ_target(s') is the target actor's deterministic action that it chooses for continuous actions.

The target network updates its gradient based on deterministic actions that the critic network perceive are apt for a state, action pair:<br /> 
∇<sub>θμ</sub>J ≈ E[∇<sub>a</sub>Q(s,a∣θ<sup>Q</sup>)∇<sub>θμ</sub>μ(s∣θ<sup>μ</sup>)]

Both DQN and DDPG use target networks (slowly updated versions of main networks) to stabilize training:

𝜃<sup>′</sup>← τθ + (1−τ)θ<sup>'</sup> with small τ

The actor is trained using the deterministic policy gradient theorem, where gradients flow from the critic's evaluation of the actor's chosen actions back through the actor network. The actor's objective is to maximize the critic's Q-value estimate: it learns to select actions that the critic believes will yield the highest returns.

The critic network learns a single Q-value for each given state-action pair for discrete actions and drives both learning and action selection. It's trained using temporal difference learning with target networks for stability. In this way it is somewhat similar to the network in the DQN algorithm, but that learns the maximal Q-values for all possible discrete state, action pairs.

## Replay Buffer

The Replay Buffer uses a queue for a replay buffer. It randomly selects past experience tuples to avoid correlation among actions and next state. It randomly samples data from the queue for more efficient sample reuse. This avoids temporal correlations during training.

## Model

The Actor and Critic neural networks used are below:

```
Actor(
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
Critic(
  (fcs1): Linear(in_features=24, out_features=256, bias=True)
  (fc2): Linear(in_features=258, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

## Plot of Rewards 

The results plot can be seen in the image results.png.

The saved weights of the Actor and Critic networks are found in the main repository.

An average reward of +0.5 over a sliding window of 100 consecutive episodes was reached after 4616 episodes. A random seed of 2 was used and the number of epsisodes can also be reduced by tuning the hyperparameters.

## Results

The agents were able to achieve an average consistent rewards of over +0.5 once training eclipsed 4616 episodes, and continued until they finished training.

## Future Improvements

It is recommended to investigate using more batch normalization in the actor and critic networks to speed up training, as well as make the agents converge faster. The training is noticably slow even when using GPUs, so speeding up training as well would prove to be fruitful in helping multi agents learn more quickly.
