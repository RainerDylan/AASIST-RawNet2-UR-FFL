import numpy as np

class DegradationSelector:
    def __init__(self):
        self.degradations = ['noise', 'quantize', 'smear', 'ripple']

    def select(self, batch_size):
        return np.random.choice(self.degradations, size=batch_size).tolist()