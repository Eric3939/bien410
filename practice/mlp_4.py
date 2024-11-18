import numpy as np
from random import random


class MLP:
    def __init__(self, input, hidden, output) -> None:
        self.input = input
        self.hidden = hidden
        self.output = output

        layers = [input]+ hidden + [output]

        w = []
        