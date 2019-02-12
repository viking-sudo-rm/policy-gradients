import torch
import numpy as np
from environment import ReverseEnvironment
import random

INPUT_SIZE = 2
STACK_SIZE = 2
ACTIONS = 2
GAMMA = 0.99
HIDDEN_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPISODES = 5000
INPUT_LENGTH = 10


class Policy(torch.nn.Module):
  def __init__(self):
    super(Policy, self).__init__()
    self.state_space = INPUT_SIZE + STACK_SIZE
    self.action_space = ACTIONS

    self.l1 = torch.nn.Linear(self.state_space, HIDDEN_SIZE, bias=False)
    self.l2 = torch.nn.Linear(HIDDEN_SIZE, self.action_space, bias=False)

    self.gamma = GAMMA

    # Episode policy and reward history
    self.policy_history = torch.autograd.Variable(torch.Tensor())
    self.reward_episode = []
    self.action_history = []

    # Overall reward and loss history
    self.reward_history = []
    self.loss_history = []

  def forward(self, x):
    model = torch.nn.Sequential(
        self.l1,
        torch.nn.Dropout(p=0.6),
        torch.nn.ReLU(),
        self.l2,
        torch.nn.Softmax(dim=-1)
    )
    return model(x)


def select_action(policy, state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
  state = state.type(torch.FloatTensor)
  state = policy(torch.autograd.Variable(state))
  c = torch.distributions.categorical.Categorical(state)
  action = c.sample()

  # Add log probability of our chosen action to our history
  if policy.policy_history.dim() != 0:
    policy.policy_history = torch.cat(
        [policy.policy_history, c.log_prob(action).unsqueeze(0)])
  else:
    policy.policy_history = (c.log_prob(action))
  return action


def update_policy(policy, optimizer):
  R = 0
  rewards = []

  # Discount future rewards back to the present using gamma
  for r in policy.reward_episode[::-1]:
    R = r + policy.gamma * R
    rewards.insert(0, R)

  # Scale rewards
  rewards = torch.FloatTensor(rewards)
  rewards = (rewards - rewards.mean()) / \
      (rewards.std() + np.finfo(np.float32).eps)

  # Calculate loss
  loss = (torch.sum(torch.mul(policy.policy_history,
                              torch.autograd.Variable(rewards)).mul(-1), -1))

  # Update network weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Save and intialize episode history counters
  policy.loss_history.append(loss.item())
  policy.reward_history.append(np.sum(policy.reward_episode))
  policy.policy_history = torch.autograd.Variable(torch.Tensor())
  policy.reward_episode = []
  policy.action_history = []


def gen_string():
  s = ""
  for i in range(INPUT_LENGTH):
    s += random.choice(["0", "1"])
  return s


def main():
  policy = Policy()
  optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
  for episode in range(NUM_EPISODES):
    env = ReverseEnvironment(gen_string())

    while not env.is_done():
      state = env.observe_next_state()
      action = select_action(policy, state)
      reward = 0
      if action == 0:
        reward = env.push()
      elif action == 1:
        reward = env.pop()

      # Save reward
      policy.reward_episode.append(reward)
      policy.action_history.append(action)

    hist = policy.action_history
    update_policy(policy, optimizer)
    if episode % 50 == 0:
      print('Episode {}\tAverage reward: {:.2f}'.format(
          episode, sum(policy.reward_history) / len(policy.reward_history)))
      print('Input:  ' + env._input_string)
      print('Output: ' + ''.join([env._tensor_to_char(x)
                                  for x in env._output_buffer]))
      print(hist)
      policy.reward_history = []


if __name__ == '__main__':
  main()
