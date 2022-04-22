import torch
import numpy as np
import matplotlib.pyplot as plt

# import relevant SWAGgy modules
from swaggy.regression_model import construct_model
from swaggy.handle_data import get_dataloaders
from swaggy.train import train, train_swag
from swaggy.swag import sample_model

def inv_standardize(element,STD,MEAN):
    return element * STD + MEAN

def mae_loss(pred, true):
    loss = np.abs(pred-true)
    return loss.mean()

# Set of params
PRESWAG = False
PRESWAG_model = '../../weights/pre-swag-resnet.pt'
swag_lr = 0.05
num_of_swagpochs = 25

dat_dir = '/users/mtoomey/scratch/shirley_project/data/vortex_axion_data'
batch_size = 64
train_test_split = 0.80
lr = 1e-3
weight_decay = 1e-4
num_of_epochs = 100


# Construct a regression model
model = construct_model()


if PRESWAG:
    # Make the dataloaders
    train_ldr, test_ldr = get_dataloaders(dat_dir,batch_size,train_test_split)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Begin regular non-swag training
    model = train(model,optimizer,train_ldr,test_ldr,num_of_epochs=num_of_epochs)
    # Save the weights
    torch.save(model.state_dict(),PRESWAG_model)
else:
    # Make the dataloaders
    train_ldr, test_ldr, stats = get_dataloaders(dat_dir,batch_size,train_test_split,return_stats=True)
    # Load in save weights from pre-swag round
    model.load_state_dict(torch.load(PRESWAG_model))
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=swag_lr)
    # Begin regular non-swag training
    model,thetaSWA, theta2, D = train_swag(model,optimizer,train_ldr,test_ldr,num_of_epochs=num_of_swagpochs)

    predicted_axion_mass_list = []
    # Sample the weights at a data point
    for i in range(250):
        # Grab a test data point
        test_image = list(test_ldr)[0][0][1]
        test_value = list(test_ldr)[0][1][1]
        new_model = sample_model(thetaSWA,theta2)
        # Use GPU if available
        if torch.cuda.is_available():
            images = test_image.cuda()
            true_mass = test_value.cuda()
            new_model = new_model.cuda().float()

        predicted_axion_mass = new_model(torch.unsqueeze(images,0))
        predicted_axion_mass_list.append(predicted_axion_mass.cpu().detach().numpy())

    predicted_axion_mass_list = np.array(predicted_axion_mass_list)

    convert_mass = inv_standardize(predicted_axion_mass_list,stats[2],stats[3])

    plt.figure(figsize=(6,6))
    plt.hist(convert_mass.flatten(),50,density=True,facecolor='g',alpha=0.75)
    plt.xlabel('Axion mass')
    plt.xlim([-25,-22])
    plt.savefig('../figures/post')

