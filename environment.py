import torch

class ReverseEnvironment:

    _NULL = torch.zeros(2)

    def __init__(self, input_string):
        self._input_string = input_string
        self._input_buffer = [self._char_to_tensor(char) for char in input_string]
        # Converting the line below to a generator will break things.
        self._input_buffer.extend([self._NULL for _ in self._input_buffer])
        self._output_buffer = []
        self._stack = []
        self._num_pops = 0

    def push(self):
        current_input = self._safe_pop(self._input_buffer)
        self._stack.insert(0, current_input)
        return 0.

    def pop(self):
        self._safe_pop(self._input_buffer)

        current_output = self._safe_pop(self._stack)
        self._output_buffer.append(current_output)
        self._num_pops += 1

        if self._num_pops > len(self._input_string):
            return 0.
        else:
            desired_output = self._input_string[-self._num_pops]
            return float(self._tensor_to_char(current_output) == desired_output)

    def observe_next_state(self):
        input_tensor = self._input_buffer[0]
        summary_tensor = self._stack[0]
        return torch.cat([input_tensor, summary_tensor])

    def is_done(self):
        return len(self._input_buffer) == 0

    @staticmethod
    def _char_to_tensor(char):
        if char == "0":
            return torch.Tensor([0, 1]).int()
        elif char == "1":
            return torch.Tensor([1, 0]).int()
        else:
            raise ValueError("Invalid character in ReverseEnvironment.")

    @staticmethod
    def _tensor_to_char(tensor):
        if tensor[0] == 1:
            return "0"
        elif tensor[1] == 1:
            return "1"

    @classmethod
    def _safe_pop(cls, poppable):
        if len(poppable) > 0:
            return poppable.pop(0)
        else:
            return cls._NONE


def test():
    env = ReverseEnvironment("10011")
    env.push()
    print("stack1", env._stack)
    env.pop()
    print("stack2", env._stack)
    print("output2", env._output_buffer)


if __name__ == "__main__":
    test()
