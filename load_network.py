import torch
import torch.nn as nn

# Define the neural network architecture
class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
             nn.Linear(3, 50),
             nn.LeakyReLU(),
             #nn.Dropout(p=0.1),
             nn.Linear(50, 200),
             nn.LeakyReLU(),
             #nn.Dropout(p=0.1),
             nn.Linear(200, 500),
             nn.LeakyReLU(),
             #nn.Dropout(p=0.1),
             nn.Linear(500, 300),
             nn.LeakyReLU(),
             nn.Linear(300, 200),
             nn.LeakyReLU(),
             nn.Linear(200, 100),
             nn.LeakyReLU(),
             #nn.Dropout(p=0.1),
             nn.Linear(100, 80),
             nn.LeakyReLU(),
             # nn.Dropout(p=0.1),
             nn.Linear(80, 30),
             nn.LeakyReLU(),
             #nn.Dropout(p=0.1),
             nn.Linear(30, 3)
        )


    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)


# Create an instance of the model
model = MLP

# Train the model and save it
#torch.save(model.state_dict(), 'my_model.pth')

# Load the saved model

model.load_state_dict(torch.load('network_checkpoint.pth'))
