import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network import Net

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
  print('mps available')
# this ensures that the current current PyTorch installation was built with MPS activated.

def read_data(model_name):
  vul_data = data_directory+model_name+'/'+model_name+'.vul' #specify the .vul file that contains the data
  with open(vul_data, 'rb') as handle:
    data = pickle.load(handle)
  return data

def ymix(species_name, data, test_data):
  '''Inputs:
      species_name: name of the species, for example 'H2O
      data: unpickled data read from the .vul output file from vulcan
    Outputs:
      y: density of the selected species, with shape the number of atmospheric vertical layers'''
  species = data['variable']['species']

  ymix = data['variable']['ymix'][:,species.index(species_name)]
  ymix_test = test_data[:,species.index(species_name)]
  return ymix, ymix_test

def plot_mixingratios(species_names, data, test_data):
   
   pressure = data['atm']['pco']*1e-6 #Convert pressure to bar
   
   plt.figure()
   for specie in species_names:
      mixingratio, test_mixingratio = ymix(specie, data, test_data)
      plt.plot(mixingratio, pressure, label=specie)
      plt.scatter(test_mixingratio.to('cpu'), pressure, label=specie+' (CNN)')
   
   plt.legend(loc='lower left')
   plt.ylim([max(pressure), min(pressure)])
   plt.xlim([1e-10, 1])
   plt.xscale('log')
   plt.yscale('log')
   plt.show()

def min_max_normalization(data):
   
   data_norm = (data-min(data))/(max(data)-min(data))
   return data_norm

def generate_training_data(model_names):
   full_training_set = []

   for model_name in model_names:
    training_data = []

    data = read_data(model_name)

    training_data = torch.Tensor(training_data)
    sflux = torch.from_numpy(data['variable']['sflux'][-1,:])
    training_data = torch.cat((training_data, sflux))
    y_ini = torch.from_numpy(data['variable']['y_ini'].flatten())
    training_data = torch.cat((training_data, y_ini))
    C_H = torch.Tensor([data['variable']['atom_ini']['C']/data['variable']['atom_ini']['H']])
    N_H = torch.Tensor([data['variable']['atom_ini']['N']/data['variable']['atom_ini']['H']])
    S_H = torch.Tensor([data['variable']['atom_ini']['S']/data['variable']['atom_ini']['H']])
    O_H = torch.Tensor([data['variable']['atom_ini']['O']/data['variable']['atom_ini']['H']])
    training_data = torch.cat((training_data, C_H))
    training_data = torch.cat((training_data, N_H))
    training_data = torch.cat((training_data, S_H))
    training_data = torch.cat((training_data, O_H))
    Tprofile = torch.from_numpy(data['atm']['Tco'])
    training_data = torch.cat((training_data, Tprofile))
    Kzz = torch.Tensor([data['atm']['Kzz'][0]])
    training_data = torch.cat((training_data, Kzz))
    gs = torch.Tensor([data['atm']['gs']])
    training_data = torch.cat((training_data, gs))

    training_data = torch.Tensor(np.float32(training_data))
    training_data = min_max_normalization(training_data)
    training_data = torch.reshape(training_data, (1, len(training_data)))
    training_data = training_data.to(DEVICE)

    target = torch.from_numpy(np.float32(data['variable']['ymix']))
    target = torch.reshape(target, (1, len(target), len(target[0])))
    target = target.to(DEVICE)

    full_training_set.append([training_data, target])
   
   return full_training_set
  
data_directory='/Users/wiebe/Documents/VULCAN_runs/HATP11/' #Directory that contains the VULCAN runs, for example '/Users/wiebe/Documents/VULCAN_runs/defined_stars/'
model_names=[]
planet_name = 'HATP11' # Name of the planet, for example 'HD189733'
Zvals = [1, 10, 30]
COvals = [0.30, 0.55, 0.70]
# Zvals = np.arange(1, 31, 1) # Values of the metallicity
# COvals = np.arange(0.1, 1, 0.1) # Values of the C/O ratio

for i in range(len(COvals)):
    for j in range(len(Zvals)):
      model_names.append(planet_name+'_Z'+str(Zvals[j])+'_CO'+format(COvals[i], '.2f')+'_T0_gp_K9_Fs')

DEVICE = torch.device('cpu') # Processing unit to use, choose from 'CPU', 'mps' (for Apple GPU), 'cuda' (for NVIDIA GPU)
print('Device: '+str(DEVICE))
learning_rate = 0.001
network = Net().to(DEVICE)  # We move the network to the GPU
n_epochs = 10

optimizer = optim.Adam(network.parameters(), lr=learning_rate)

full_training_set = generate_training_data(model_names)

test_model = ['HATP11_Z30_CO0.30_T0_gp_K9_Fs'] # List of test models, for example ['HD189_Z10_CO0.3_T0_gp_K9_Fs']
test_set = generate_training_data(test_model)#[full_training_set[0]] 

train_losses = []
train_counter = []
test_losses = []

test_counter = [i*len(full_training_set) for i in range(n_epochs + 1)]

" This is the main training loop "
log_interval = 1
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(full_training_set):
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(full_training_set),
                100. * batch_idx / len(full_training_set), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx) + ((epoch-1)*len(full_training_set)))

" This is the main testing loop "
def test():
    network.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_set:
            output = network(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
    test_loss /= len(test_set)
    test_losses.append(test_loss)

    return output

" Let's do the training "
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)

" Here we plot the loss curve "
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

" Here we plot the output mixing ratios"
data = read_data(test_model[0]) # True value of the mixingratios
test_data = test()[0] # Output of the neural network

species_names = ['H2O', 'SO2'] #List of the species to plot, for example ['H2S', 'SO2', 'CO2']

plot_mixingratios(species_names, data, test_data)
