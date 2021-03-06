import torch
import numpy as np

from swaggy.regression_model import Resnet18Regression

def track_SWAG_params(model,thetaSWA,theta2,SWAGpoch):
    # Don't track any changes to the weights
    with torch.no_grad():
        nn = []
        for param in model.parameters():
            # Convert weights of each layer to numpy array
            param_numpy = param.detach().numpy()
            nn.append(param_numpy)
        # Convert python list of numpy arrays to numpy array
        nn = np.array(nn,dtype=object)

        if SWAGpoch == 0:
          # Update our running sum
          thetaSWA = nn
          # placeholder for theta2
          theta2 = 0 * nn 
        else:
          # Calculating theta_SWA: thetaSWA = 1/T Sum_i theta_i
          thetaSWA = 1./(SWAGpoch + 1) * ((SWAGpoch)*thetaSWA + nn)
          # Calculating bar{theta}^2: \bar{\theta}^2 = 1/T Sum_i theta_i^2
          theta2 = 1./(SWAGpoch + 1) * (np.power(nn,2) + (SWAGpoch)*theta2)

    return thetaSWA, theta2


def sample_model(thetaSWA,theta2):
  model_swag = Resnet18Regression(1,1).to("cpu")

  # Loop trough copy of model, grab new weights stochastically
  with torch.no_grad():
    for param, tSWA, t2 in zip(model_swag.parameters(), thetaSWA, theta2):
      mean, sigma = tSWA, np.sqrt(t2)
      param.data = torch.from_numpy(np.random.normal(mean,sigma))

  return model_swag
