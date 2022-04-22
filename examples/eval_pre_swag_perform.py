import torch
import matplotlib.pyplot as plt
import numpy as np

# load swaggy stuff
from swaggy.regression_model import construct_model
from swaggy.handle_data import get_dataloaders

# Mean Absolute Error is used as the main metric for measuring the performance of the model.
def mae_loss(pred, true):
    loss = np.abs(pred-true)
    return loss.mean()

def inv_standardize(element,STD,MEAN):
    return element * STD + MEAN

# Construct a regression model
PRESWAG_model = '../../weights/pre-swag-resnet.pt'
dat_dir = '/users/mtoomey/scratch/shirley_project/data/vortex_axion_data'
batch_size = 64
train_test_split = 0.80

# Make the dataloaders
train_ldr, test_ldr,stats = get_dataloaders(dat_dir,batch_size,train_test_split,return_stats=True)

# Construct a regression model
model = construct_model()
model.load_state_dict(torch.load(PRESWAG_model))
model.eval()

# Run the model on the test dataset
predicted_axion_mass_list = []
real_axion_mass_list = []
for step, (images, axion_mass) in enumerate(test_ldr):
    # Use GPU if available
    if torch.cuda.is_available():
      images = images.cuda()
      axion_mass = axion_mass.cuda()
      model = model.cuda()

    # RUN the model
    predicted_axion_mass = model(images)
    predicted_axion_mass_list.append(predicted_axion_mass.cpu().detach().numpy())
    real_axion_mass_list.append(axion_mass.cpu().numpy())


# Remove the last batch of the results to make all the arrays in the list the same size
del predicted_axion_mass_list[-1]
del real_axion_mass_list[-1]


# PLOTTING TEST RESULTS
predicted_axion_mass_arr = np.concatenate(predicted_axion_mass_list)
real_axion_mass_arr = np.concatenate(real_axion_mass_list)

m_pred,m_true = inv_standardize(predicted_axion_mass_arr,stats[2],stats[3]),inv_standardize(real_axion_mass_arr,stats[2],stats[3])

test_mae = mae_loss(m_pred,m_true)
plt.figure(figsize=(8,8),dpi=80)
plt.scatter(m_true, m_pred,  color='black')
line = np.linspace(-24, -22, 10)
plt.plot(line, line)
plt.xlabel('Observed Axion Mass')
plt.ylabel('Predicted Axion Mass')
plt.ylim([-24,-22])
plt.xlim([-24,-22])
plt.savefig('../figures/eval_pre_swag')

print(f'MAE: {test_mae}')
