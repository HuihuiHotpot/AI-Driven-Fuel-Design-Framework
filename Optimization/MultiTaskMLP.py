from torch import nn

class MultiTaskMLP(nn.Module):
    def __init__(self, input_size, shared_layer_sizes, task1_sizes, task2_sizes, output_sizes):
        super(MultiTaskMLP, self).__init__()

        shared_layers = []
        for i, size in enumerate(shared_layer_sizes):
            shared_layers.append(nn.Linear(input_size if i == 0 else shared_layer_sizes[i - 1], size))
            shared_layers.append(nn.ReLU())
        self.shared_layers = nn.Sequential(*shared_layers)

        task1_layers = []
        for i, size in enumerate(task1_sizes):
            task1_layers.append(nn.Linear(shared_layer_sizes[-1] if i == 0 else task1_sizes[i - 1], size))
            task1_layers.append(nn.ReLU())
        self.task1_layers = nn.Sequential(*task1_layers)
        self.task1_output = nn.Linear(task1_sizes[-1], output_sizes[0])

        task2_layers = []
        for i, size in enumerate(task2_sizes):
            task2_layers.append(nn.Linear(shared_layer_sizes[-1] if i == 0 else task2_sizes[i - 1], size))
            task2_layers.append(nn.ReLU())
        self.task2_layers = nn.Sequential(*task2_layers)
        self.task2_output = nn.Linear(task2_sizes[-1], output_sizes[1])

    def forward(self, x):
        shared_features = self.shared_layers(x)
        task1_features = self.task1_layers(shared_features)
        task2_features = self.task2_layers(shared_features)
        output1 = self.task1_output(task1_features)
        output2 = self.task2_output(task2_features)
        return output1, output2