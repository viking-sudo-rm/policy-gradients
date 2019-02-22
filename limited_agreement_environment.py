class LimitedAgreementEnvironment:

  def __init__(self, sentence, labels):
    self._sentence = list(sentence)
    self._output = []
    self._labels = list(labels)
    self._char_i = -1
    self._stack = []
    self.actions = self._make_actions()

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
    # return -100, True

  # def _make_output_action(self, value):

  #   def _output_action():
  #     return float(value == self._label), True

  #   return _output_action

  def _output_action(self):
    # if self._char_i != len(self._sentence) - 1:
    #   return -100, True
    if len(self._stack) > 0:
      self._output.append(self._stack.pop(0))

      reward = float(self._output[-1] == self._labels[self._char_i])
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

  # def _are_chars_exhausted(self):
  #     # For now, match inputs to actions. Don't need to do this in principle.
  #     return self._char_i >= len(self._sentence)
