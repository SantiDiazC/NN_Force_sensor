import scipy.io
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

# Load the Matlab file
mat_file = scipy.io.loadmat('Sensor1.mat')

# Access the variables in the Matlab file
input = mat_file['S_learn']
target = mat_file['F_learn']


input = np.asarray(np.transpose(input))
target = np.asarray(np.transpose(target))*1000

print(input.shape)
print(target.shape)

class MyDataset(torch.utils.data.Dataset):
    '''
    Prepare the Boston dataset for regression
    '''

    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# X, y = load_boston(return_X_y=True)
dataset = MyDataset(input, target)

# Split the dataset into train, validation, and test sets
train_size = int(len(dataset) * 0.8)
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f'Train dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

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

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for train, validation, and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# Initialize the MLP
mlp = MLP()
mlp.train()

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

# Set fixed random number seed
torch.manual_seed(42)
wb_train_loss = 0.0
wb_val_loss = 0.0
# Train loop
for epoch in range(0, 400):  # 5 epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch + 1}')

    print('Learning rate :%.5f' %(optimizer.param_groups[0]['lr']))

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(train_loader, 0):

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 3))

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()

        if i % 1000 == 0:
            print('Loss after mini-batch %5d: %.6f' %
                  (i + 1, current_loss / 1000))
            wb_train_loss = current_loss / 1000
            current_loss = 0.0

    scheduler.step()

    # Validate loop
    mlp.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 3))
            outputs = mlp(inputs)
            val_loss += loss_function(outputs, targets).item()

        # Print validation statistics
        print('Validation loss: %.6f' % (val_loss / len(val_loader)))
        wb_val_loss = val_loss / len(val_loader)
    print("Train loss: ", wb_train_loss)
    print("Val loss: ", wb_val_loss)
    mlp.train()

# Test loop
mlp.eval()
with torch.no_grad():
    test_loss = 0.0
    for i, data in enumerate(test_loader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 3))
        outputs = mlp(inputs)
        test_loss += loss_function(outputs, targets).item()

    # Print test statistics
    print('Test loss: %.6f' % (test_loss / len(test_loader)))