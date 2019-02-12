import torch
import numpy as np
from environment import ReverseEnvironment
import random

INPUT_SIZE = 2
STACK_SIZE = 2
ACTIONS = 2
GAMMA = 1
HIDDEN_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPISODES = 5000
INPUT_LENGTH = 10
BATCH_SIZE = 10


class Policy(torch.nn.Module):
  def __init__(self):
    super(Policy, self).__init__()
    self.state_space = INPUT_SIZE + STACK_SIZE
    self.action_space = ACTIONS

    self.l1 = torch.nn.Linear(self.state_space, HIDDEN_SIZE)
    self.l2 = torch.nn.Linear(HIDDEN_SIZE, self.action_space)

    self.gamma = GAMMA

    # Episode policy and reward history
    self.policy_history = []
    self.reward_batch = []
    self.action_history = []

    # Overall reward and loss history
    self.reward_history = []

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
  # print(state)
  action = c.sample()

  # Add log probability of our chosen action to our history
  if policy.policy_history[-1].dim() != 0:
    policy.policy_history[-1] = torch.cat(
        [policy.policy_history[-1], c.log_prob(action).unsqueeze(0)])
  else:
    policy.policy_history[-1] = (c.log_prob(action))
  return action


def update_policy(policy, optimizer):
  rewards = []

  # Discount future rewards back to the present using gamma
  for i in range(len(policy.reward_batch)):
    rewards.append([])
    R = 0
    for r in policy.reward_batch[i][::-1]:
      R = r + policy.gamma * R
      rewards[i].insert(0, R)

  # Scale rewards
  rewards = torch.FloatTensor(rewards)
  rewards = (rewards - rewards.mean()) / \
      (rewards.std() + np.finfo(np.float32).eps)

  # Calculate loss
  # loss = (torch.sum(torch.matmul(torch.stack(policy.policy_history, dim=0),
                                #  torch.autograd.Variable(torch.transpose(rewards, 0, 1))).mul(-1), -1))

  loss = torch.sum((torch.stack(policy.policy_history, dim=0) * torch.autograd.Variable(rewards)).mul(-1))
  # print(policy.policy_history)
  # print(torch.stack(policy.policy_history, dim=0).shape)
  # print(torch.transpose(rewards, 0, 1).shape)
  # print(loss.shape)

  # Update network weights
  optimizer.zero_grad()
  loss.backward()
  # norm = np.mean([torch.mean(param.grad) for param in policy.parameters()])
  optimizer.step()

  # Save and intialize episode history counters
  for i in range(len(policy.reward_batch)):
    policy.reward_history.append(np.sum(policy.reward_batch[i]))
  policy.policy_history = []
  policy.reward_batch = []
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
    policy.policy_history.append(torch.autograd.Variable(torch.Tensor()))
    policy.reward_batch.append([])
    policy.action_history.append([])
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
      policy.reward_batch[-1].append(reward)
      policy.action_history[-1].append(action)

    if episode % 10 != 0:
      continue

    policy_hist = policy.policy_history
    action_hist = policy.action_history
    update_policy(policy, optimizer)
    if episode % 50 == 0:
      print('Episode {}\tAverage reward: {:.2f}'.format(
          episode, np.sum(policy.reward_history) / len(policy.reward_history)))
      print('Input:  ' + env._input_string)
      print('Output: ' + ''.join([env._tensor_to_char(x)
                                  for x in env._output_buffer]))
      # print(policy_hist)
      # print(action_hist)
      policy.reward_history = []


if __name__ == '__main__':
  main()
