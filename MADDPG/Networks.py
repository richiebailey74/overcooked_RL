import torch


class DeepCriticNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepCriticNetwork, self).__init__()
        # each call to Linear automatically initializes the weights and biases
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, output_size)

        # set the activation function to be used in forward()
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.ReLU(self.fc3(x))
        x = self.fc4(x)  # output layer needs raw neuron value, no ReLU needed
        return x


class DeepActorNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepActorNetwork, self).__init__()
        # each call to Linear automatically initializes the weights and biases
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, output_size)

        # set the activation function to be used in forward()
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.ReLU(self.fc3(x))
        # add a batch dimension for when sampling to get next action (or else will get dimension errors)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.softmax(self.fc4(x))  # output layer needs to be a probability distribution to sample from
        return x
