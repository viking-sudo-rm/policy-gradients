import torch
import numpy as np
import random

from agreement_environment import LinzenDataset
from limited_agreement_environment import LimitedAgreementDataset


VOCABULARY_SIZE = 12000
EMBEDDING_SIZE = 50
HIDDEN_SIZE = 128
GAMMA = 1
LEARNING_RATE = 0.01
NUM_EPISODES = 1000000
BATCH_SIZE = 128


class Policy(torch.nn.Module):
  def __init__(self, num_actions, lstm_on=False):
    super(Policy, self).__init__()

    self._lstm_on = lstm_on
    self.embedding = torch.nn.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE)

    if self._lstm_on:
      self.lstm_cell = torch.nn.LSTMCell(EMBEDDING_SIZE + 2, HIDDEN_SIZE)
      self.h, self.c = None, None
    else:
      self.l1 = torch.nn.Linear(EMBEDDING_SIZE + 2, HIDDEN_SIZE)
    self.l2 = torch.nn.Linear(HIDDEN_SIZE, num_actions)

    self.gamma = GAMMA

    # Episode policy and reward history
    self.policy_history = []
    self.reward_batch = []
    self.action_history = []
    self.accuracy_history = []

    # Overall reward and loss history
    self.reward_history = []

  def forward(self, word, stack):
    embedded = self.embedding(word)
    observation = torch.cat([embedded, stack])
    if self._lstm_on:
      self.h, self.c = self.lstm_cell(observation.unsqueeze(0), [self.h, self.c])
      output1 = self.h.squeeze(0)
    else:
      output1 = torch.relu(self.l1(observation))
    output2 = torch.softmax(self.l2(output1), -1)
    return output2

  def init_state(self):
    if self._lstm_on:
      self.h = torch.zeros(1, HIDDEN_SIZE)
      self.c = torch.zeros(1, HIDDEN_SIZE)


def select_action(policy, state):
  # Select an action (0 or 1) by running policy model and choosing based on the
  # probabilities in state
  # state = state.type(torch.FloatTensor)
  word_var = torch.autograd.Variable(torch.tensor(state[0]))
  stack = torch.zeros(2)
  if state[1] is not None:
    stack[state[1]] = 1
  stack_var = torch.autograd.Variable(torch.tensor(stack))
  # print('word', str(word_var))
  # print('stack', str(stack_var))
  state = policy(word_var, stack_var)

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

  # print("reward history", policy.reward_history)

  # Discount future rewards back to the present using gamma
  for i in range(len(policy.reward_batch)):
    rewards.append([])
    R = 0
    for r in policy.reward_batch[i][::-1]:
      R = r + policy.gamma * R
      rewards[i].insert(0, R)

  max_length = max(len(reward_seq) for reward_seq in rewards)
  for i, reward_seq in enumerate(rewards):
    num_missing = max_length - len(reward_seq)
    this_length = len(reward_seq)
    reward_seq.extend(0. for _ in range(num_missing))
    padded_history = torch.zeros(max_length)
    padded_history[:this_length] = policy.policy_history[i]
    policy.policy_history[i] = padded_history

  # print("rewards", rewards)
  # print("action history", policy.action_history)

  # Scale rewards
  rewards = torch.FloatTensor(rewards)
  # TODO:  Hmm.. should this be row-wise mean???
  rewards = (rewards - rewards.mean()) / \
      (rewards.std() + np.finfo(np.float32).eps)

  # print("policy history", policy.policy_history)

  loss = torch.sum((torch.stack(policy.policy_history, dim=0) * torch.autograd.Variable(rewards)).mul(-1))

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
  policy.accuracy_history = []

def main():

  dataset = LinzenDataset()

  rewards = []
  policy = Policy(len(dataset.get_env().actions), lstm_on=True)
  optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

  for episode in range(NUM_EPISODES):
    policy.policy_history.append(torch.autograd.Variable(torch.Tensor()))
    policy.reward_batch.append([])
    policy.action_history.append([])

    env = dataset.get_env()

    policy.init_state()
    done = False
    while not done:
      state = env.observe_environment()
      if state[0] is None:
        break

      action = select_action(policy, state)
      reward, done = env.actions[action]()

      # Save reward
      policy.reward_batch[-1].append(reward)
      policy.action_history[-1].append(action)
      policy.accuracy_history.append(env.accuracy)

    reward = np.sum(policy.reward_batch[-1])
    rewards.append(reward)

    if episode % BATCH_SIZE != BATCH_SIZE - 1:
      continue

    action_history = policy.action_history
    mean_accuracy = np.mean(policy.accuracy_history)
    update_policy(policy, optimizer)
    mean_reward = np.mean(policy.reward_history)
    policy.reward_history = []

    print("=" * 50)
    print('Episode {}\tAverage reward: {:.2f}\tAccuracy: {:.2f}'.format(episode, mean_reward, mean_accuracy))
    # print('Input:', ' '.join(token.text for token in instance["sentence"]))
    print("Input:", dataset.input_string)
    print("Output (correct)", env.output)
    print("Action History:", [t.item() for t in action_history[-1]])
    #print('Output / Label:', env._output, "/", env._label)

  import matplotlib.pyplot as plt
  plt.plot([max(0, reward) for reward in rewards])
  plt.show()


if __name__ == '__main__':
  main()
