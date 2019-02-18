class LinzenEnvironment:

    def __init__(self, sentence, label):
        self._sentence = sentence
        self._label = label
        self._char_i = -1
        self._stack = []
        self.actions = self._make_actions()


    """Generic declarations for types of actions."""

    def _make_push_action(self, value):

        def _push_action():
            self._stack.insert(0, value)
            return 0., False

        return _push_action

    def _make_pop_action(self, value):

        def _pop_action():
            if len(self._stack) > 0:
                self._stack.pop(0)
                return 0., False
            return -float("inf"), True

        return _pop_action

    def _make_swap_action(self, value):

        def _swap_action():
            if len(self._stack) > 0:
                self._stack[0] = 1 - self._stack[0]
                return 0., False
            return -float("inf"), True

        return _swap_action

    def _make_output_action(self, value):
        def _output_action():
            return float(value == label), True


    """Specific mapping of integers to actions."""

    def _make_actions(self):
        return [
            self._make_push_action(0),
            self._make_push_action(1),
            self._make_pop_action(0),
            self._make_pop_action(1),
            self._make_swap_action(0),
            self._make_swap_action(1),
            self._make_output_action(0),
            self._make_output_action(1),
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

    # def _are_chars_exhausted(self):
    #     # For now, match inputs to actions. Don't need to do this in principle.
    #     return self._char_i >= len(self._sentence)
