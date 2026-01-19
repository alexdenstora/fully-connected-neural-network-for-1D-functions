import torch
from torch import nn

import matplotlib.pyplot as plt

######################################
# Q1 Custom Dataset for 1D Function
######################################
class SimpleCurveData(torch.utils.data.Dataset):

  def __init__(self):
    self.x_data = torch.arange(-2.5,2.5,0.001).unsqueeze(1)
    self.y_data = torch.max(2 * self.x_data, 3 * torch.sin(5 * self.x_data) + self.x_data ** 2) #TODO: set y = the computation of f(x) = max(2x, 3sin(5x) + x^2)

  def __len__(self):
    return self.x_data.shape[0]

  def __getitem__(self, i):
    #TODO
    return self.x_data[i], self.y_data[i]

######################################
# Q2 Feed Forward Neural Network
######################################

class FeedForwardNetwork(nn.Module):

  def __init__(self, input_dim, layer_widths, output_dim, activation=nn.ReLU):
    super().__init__()
    self.layer_widths = layer_widths
    assert(len(layer_widths) >= 1)

    #TODO
    # create a list to store the network layers
    self.layers = nn.ModuleList()
    # for each layer width i initialize and append a linear transformation as well as an activation function to layers
    self.layers.append(nn.Linear(input_dim, layer_widths[0]))
    for i in range(1, len(layer_widths) - 1): # skip first and last layer
      self.layers.append(activation()) # append a ReLU
      self.layers.append(nn.Linear(layer_widths[i - 1], layer_widths[i]))

    # append another ReLU Linear layer
    self.layers.append(activation())
    self.layers.append(nn.Linear(layer_widths[-1], output_dim))

  def forward(self, x):
    #TODO feed the input through the layers and return the output
    out = x
    for i in range(len(self.layers)):
      out = self.layers[i](out)
    return out

  def getParamCount(self):
    return sum(p.numel() for p in self.parameters())

######################################
# Q3 Training Loop
######################################

def train(model, dataloader):
  # TODO
  # instantiate optimizer and loss functions
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
  loss_fn = torch.nn.MSELoss()

  max_epochs = 99
  for i in range(max_epochs):
    if i % 20 == 0: # displaying predictions to visualize model
      plotPredictions(model, dataloader, i)

    for batch in dataloader:
      optimizer.zero_grad() # zero gradients
      inputs,targets = batch # unpack batch
      outputs = model(inputs) # run batch through model
      loss = loss_fn(outputs, targets) # calculate loss
      loss.backward() # backpropagate gradient of the loss to model params
      optimizer.step() # take a step of the optimizer

  plotPredictions(model, dataloader, max_epochs) # plot final epoch results

      


def plotPredictions(model, dataloader, epoch, save_fig=False):
  dataset = dataloader.dataset
  # Switch to eval mode
  model.eval()

  # Make predictions for the full dataset
  y_pred = model(dataset.x_data)

  # Compute loss for reporting
  loss = ((dataset.y_data-y_pred)**2).mean().item()

  # Plot dataset and predictions
  plt.figure(figsize=(10,4))
  plt.plot(dataset.x_data.data, dataset.y_data.data, color="black", linestyle='dashed', label="True Function")
  plt.plot(dataset.x_data.data, y_pred.data, color="red", label="Network")
  plt.text(-2.6,-2.3, "Epoch: {}".format(epoch))
  plt.text(-2.6,-3.0, "Loss: {:0.3f}".format(loss))
  plt.text(-2.6,-3.7, "Params: {}  {}".format(model.getParamCount(), str(model.layer_widths)))
  plt.ylim(-4,10)

  if save_fig:
    plt.savefig("epoch{}.png".format(epoch))
  else:
    plt.show()

  model.train()


if __name__ == "__main__":
    # Generate a dataset for our curve
    dataset = SimpleCurveData()

    # Set up a dataloader to grab batches
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


    # Sanity Check: Visualize the whole dataset as a curve and a batch as points
    x_batch, y_batch = next(iter(dataloader))

    plt.figure(figsize=(10,4))
    plt.plot(dataset.x_data.data, dataset.y_data.data, color="black", linestyle='dashed', label="True Function")
    plt.scatter(x_batch, y_batch, c="red", s=40, label="Batch Samples",alpha=0.75, edgecolors='none')
    plt.legend()
    plt.title("Sanity Check: Curve Dataset and Dataloader")
    plt.show()


    ######################################
    # Q4 through Q7 Experimentation
    ######################################

    # Build our network
    layer_widths = [64]
    model = FeedForwardNetwork(1, layer_widths, 1, activation=nn.ReLU)

    train(model, dataloader)