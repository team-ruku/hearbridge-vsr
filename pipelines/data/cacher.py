import torch


class ContextCacher:
    def __init__(self, buffer_size: int, context_length: int) -> None:
        self.segment_length = buffer_size
        self.context_length = context_length

        self.context = torch.zeros([self.context_length, 3, 340, 600])

    def __call__(self, chunk):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(
                chunk, (0, 0, 0, 0, 0, 0, 0, self.segment_length - chunk.size(0))
            )

        if self.context_length == 0:
            return chunk.float()

        else:
            chunk_with_context = torch.cat((self.context, chunk))
            self.context = chunk[-self.context_length :]
            return chunk_with_context.float()
