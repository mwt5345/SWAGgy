import torch

# import relevant SWAGgy modules
from swaggy.regression_model import construct_model
from swaggy.handle_data import get_dataloaders
from swaggy.train import train, train_swag

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
	# Load in save weights from pre-swag round
    model.load_state_dict(torch.load(PRESWAG_model))
	# Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=swag_lr)
    # Begin regular non-swag training
    model,thetaSWA, theta2, D = train_swag(model,optimizer,train_ldr,test_ldr,num_of_epochs=num_of_swagpochs)
