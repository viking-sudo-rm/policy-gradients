import nltk_wrapper
import random

class LimitedAgreementEnvironment:

  def __init__(self, sentence, labels, ignore_middle_verbs=True):
    self._sentence = list(sentence)
    self._output = []
    self._labels = list(labels)
    self._char_i = -1
    self._stack = []
    self.actions = self._make_actions()
    self._ignore_middle_verbs = ignore_middle_verbs

  @property
  def output(self):
    return " ".join(str(char) for char in self._output)

  """Generic declarations for types of actions."""

  def _pass_action(self):
    return 0., self._is_done()

  def _make_push_action(self, value):

    def _push_action():
      self._stack.insert(0, value)
      return 0, self._is_done()

    return _push_action

  def _pop_action(self):
    if len(self._stack) > 0:
      self._stack.pop(0)
      return 0., self._is_done()
    return 0, self._is_done()
    # return -100, True

  def _swap_action(self):
    if len(self._stack) > 0:
      self._stack[0] = 1 - self._stack[0]
      return 0., self._is_done()
    return 0, self._is_done()

  def _output_action(self):
    if len(self._stack) > 0:
      self._output.append(self._stack.pop(0))

      should_ignore = self._ignore_middle_verbs and self._char_i != len(self._sentence) - 1
      reward = float(self._output[-1] == self._labels[self._char_i]
                     and not should_ignore)
      return reward, self._is_done()
    else:
      return 0., self._is_done()

  """Specific mapping of integers to actions."""

  def _make_actions(self):
    return [
        self._make_push_action(0),
        self._make_push_action(1),
        self._pop_action,
        # self._pass_action,
        # self._swap_action,
        self._output_action,
    ]

  """Getting new inputs."""

  def observe_environment(self):
    """Get an incomplete observation of the input and stack state.

    Returns:
        Next input token.
        The top value on the stack.
    """
    return self._observe_next_char(), self._observe_stack()

  def _observe_next_char(self):
    """Return next character or None if we have reached the end of the sentence."""
    self._char_i += 1
    if self._char_i < len(self._sentence):
      return self._sentence[self._char_i]

  def _observe_stack(self):
    """Return 0, 1, or None depending on the top of the stack."""
    if len(self._stack) > 0:
      return self._stack[0]

  def _is_done(self):
    return self._char_i == len(self._sentence) - 1

class LimitedAgreementDataset:
    def __init__(self, ignore_middle_verbs=True, depth=5):
        self.grammar = nltk_wrapper.load_grammar("grammars/simple-agreement.grammar")
        self.sents = list(nltk_wrapper.generate(self.grammar, depth=depth))
        self.x_sent = None
        self.y_sent = None
        self.x_sents = [[self.filter_x(word) for word in sent] for sent in self.sents]
        self.y_sents = [[self.filter_y(word) for word in sent] for sent in self.sents]
        self._ignore_middle_verbs = ignore_middle_verbs

    def get_env(self):
        idx = random.randint(0, len(self.x_sents) - 1)
        self.x_sent, self.y_sent = self.x_sents[idx], self.y_sents[idx]
        return LimitedAgreementEnvironment(self.x_sent, self.y_sent, self._ignore_middle_verbs)

    @property
    def input_string(self):
        return " ".join(str(char) for char in self.x_sent)

    @property
    def label(self):
        return " ".join(str(char) for char in self.y_sent)

    def filter_x(self, word):
      if word == "N0":
        return 0
      elif word == "N1":
        return 1
      else:
        return 2

    def filter_y(self, word):
      if word == "V0":
        return 0
      elif word == "V1":
        return 1
      else:
        return 2
