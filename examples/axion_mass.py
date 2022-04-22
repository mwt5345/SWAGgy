import torch

# import relevant SWAGgy modules
from swaggy.regression_model import construct_model
from swaggy.handle_data import get_dataloaders
from swaggy.train import train

# Set of params
PRESWAG = False
PRESWAG_model = '../weights/pre-swag-resnet.pt'
dat_dir = '/users/mtoomey/scratch/shirley_project/data/vortex_axion_data'
batch_size = 64
train_test_split = 0.80
lr = 1e-3
weight_decay = 1e-4
num_of_epochs = 100

# Make the dataloaders
train_ldr, test_ldr = get_dataloaders(dat_dir,batch_size,train_test_split)

# Construct a regression model
model = construct_model()


if PRESWAG:
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Begin regular non-swag training
    model = train(model,optimizer,train_ldr,test_ldr,num_of_epochs=num_of_epochs)
    # Save the weights
    torch.save(model.state_dict(),PRESWAG_model)
else:
    model.load_state_dict(torch.load(PRESWAG_model))


