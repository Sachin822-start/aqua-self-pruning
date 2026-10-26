
from nn.layers import Linear


class MLP:
    """A small multilayer perceptron for classification."""

    def __init__(self, input_size, hidden_size, num_classes, seed=42):
        self.layer1 = Linear(input_size, hidden_size, seed=seed)
        self.layer2 = Linear(hidden_size, num_classes, seed=seed + 1)

    def forward(self, x):
        hidden = self.layer1.forward(x).relu()
        logits = self.layer2.forward(hidden)
        return logits

    def parameters(self):
        return (
            self.layer1.parameters()
            + self.layer2.parameters()
        )

    def layers(self):
        return [self.layer1, self.layer2]
