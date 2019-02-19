import torch
import numpy as np
from agreement_environment import LinzenEnvironment
import random
from sklearn.feature_extraction.text import CountVectorizer

VOCABULARY_SIZE = 10000
EMBEDDING_SIZE = 50
NUM_ACTIONS = len(LinzenEnvironment('', 0).actions)
GAMMA = 1
HIDDEN_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPISODES = 5000
INPUT_LENGTH = 10
BATCH_SIZE = 10


class Policy(torch.nn.Module):
  def __init__(self):
    super(Policy, self).__init__()

    self.embedding = torch.nn.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE)
    self.l1 = torch.nn.Linear(EMBEDDING_SIZE, HIDDEN_SIZE)
    self.l2 = torch.nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)

    self.gamma = GAMMA

    # Episode policy and reward history
    self.policy_history = []
    self.reward_batch = []
    self.action_history = []

    # Overall reward and loss history
    self.reward_history = []

  def forward(self, word, stack):
    embedded = self.embedding(word)
    output1 = torch.relu(self.l1(torch.cat(embedded, stack)))
    output2 = torch.softmax(self.l2(output1))
    return output2


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



def main():
  policy = Policy()
  optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
  train_data = open('data/rnn_agr_simple/numpred.train', 'r').readlines()
  X = [sent[4:] for sent in train_data]
  Y = [0 if sent[:3] == 'VBZ' else 1 for sent in train_data]
  vectorizer = CountVectorizer()
  vectorizer.fit(X)

  for episode in range(NUM_EPISODES):
    policy.policy_history.append(torch.autograd.Variable(torch.Tensor()))
    policy.reward_batch.append([])
    policy.action_history.append([])

    idx = random.randint(0, len(X) - 1)
    sent = vectorizer.transform(X[idx])
    label = Y[idx]

    env = LinzenEnvironment(sent, label)

    done = False
    while not done:
      state = env.observe_environment()

      action = select_action(policy, state)
      reward, done = env.actions[action]()

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
      # print('Input:  ' + sent)
      # print('Output: ' + ''.join([env._tensor_to_char(x)
      #                             for x in env._output_buffer]))
      # print(policy_hist)
      # print(action_hist)
      policy.reward_history = []


if __name__ == '__main__':
  main()
