from imports import nn

class SiameseNetworkTriplet(nn.Module):
    def __init__(self, input_size, hidden_size1=256):
        super(SiameseNetworkTriplet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            # nn.Linear(hidden_size1, hidden_size2),
            # nn.ReLU(),
        )
    def forward_sequential(self, x):
        return self.fc(x)

    def forward(self, input1, input2, input3):
        output1 = self.forward_sequential(input1)
        output2 = self.forward_sequential(input2)
        output3 = self.forward_sequential(input3)
        return output1, output2, output3
    
    
class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1=256):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.ReLU()

        )
    def forward_sequential(self, x):
        return self.fc(x)

    def forward(self, input1, input2):
        output1 = self.forward_sequential(input1)
        output2 = self.forward_sequential(input2)
        return output1, output2



class SiameseClassificationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, num_classes=2):
        super(SiameseClassificationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            # nn.Linear(hidden_size1, hidden_size2),
            # nn.ReLU(),
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(input_size, num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward_sequential(self, x):
        return self.fc(x)

    def forward(self, input1, input2):

        output1 = self.forward_sequential(input1)
        output2 = self.forward_sequential(input2)
        output3 = self.sm(self.fc1(output1))


        return output1, output2, output3



