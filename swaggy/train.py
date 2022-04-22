import torch
import copy

from swaggy.swag import track_SWAG_params

def mse_loss(pred, true):
    loss = (pred-true).pow(2)
    return loss.mean()

def train_swag(model,optimizer,train_data_loader,test_data_loader,scheduler=None,num_of_epochs=1):
    print('----------------------------------------------------')
    print('Begin training with SWAG')
    print('----------------------------------------------------\n')
    print('\tTraining set-up:')
    print('\t----------------\n')
    print('\tNumber of SWAGpochs: ' + str(num_of_epochs) + '\n')

    print(f'\t\t\tTrain\t\tValidation')
    print(f'\t--------------------------------------------')
    # Start the training loop
    for i in range(num_of_epochs):
        epoch_loss, val_epoch_loss = 0, 0
        num_of_steps_in_epoch, val_num_of_steps_in_epoch = 0, 0

        for step, (images_batch, axion_mass_batch) in enumerate(train_data_loader):
            optimizer.zero_grad()

            # Use GPU if available
            if torch.cuda.is_available():
                images_batch = images_batch.cuda()
                axion_mass_batch = axion_mass_batch.cuda()

            # RUN the model
            predicted_axion_mass = model(images_batch)
        
            # Calculate loss
            loss = mse_loss(predicted_axion_mass,axion_mass_batch)

            # Calculate gradient
            loss.backward()

            # Do an optimization step
            optimizer.step()

            ########## SWAGGY STUFF ##############
            # Make copy of model and pass to cpu
            model_copy = copy.deepcopy(model); model_copy.to('cpu')
            # Initialize on the first pass
            if i == 0:
              thetaSWA, theta2 = track_SWAG_params(model_copy,None,None,0)
            else:
              thetaSWA, theta2 = track_SWAG_params(model_copy,thetaSWA,theta2,i)
            ########### SWAGGY END ###############

            epoch_loss+=loss
            num_of_steps_in_epoch+=1

        loss_w = round((epoch_loss/num_of_steps_in_epoch).detach().item(),5)

        with torch.no_grad():
            for step, (images_batch, axion_mass_batch) in enumerate(test_data_loader):
                # Use GPU if available
                if torch.cuda.is_available():
                    images_batch = images_batch.cuda()
                    axion_mass_batch = axion_mass_batch.cuda()

                # RUN the model
                predicted_axion_mass = model(images_batch)
            
                # Calculate loss
                val_loss = mse_loss(predicted_axion_mass,axion_mass_batch)

                val_epoch_loss+=val_loss
                val_num_of_steps_in_epoch+=1

        val_loss_w = round((val_epoch_loss/val_num_of_steps_in_epoch).detach().item(),5)

        if (i + 1) % 1 == 0:
            print(f'\tSWAGpoch {i+1} loss:\t{loss_w}\t\t{val_loss_w}')

    print('\n\tDONE.\n')
    print('----------------------------------------------------')

    # TODO: Add-in calculation of D
    D = None

    return model, thetaSWA, theta2, D


def train(model,optimizer,train_data_loader,test_data_loader,scheduler=None,num_of_epochs=1):
    print('----------------------------------------------------')
    print('Begin Non-SWAG training')
    print('----------------------------------------------------\n')
    print('\tTraining set-up:')
    print('\t----------------\n')
    print('\tNumber of epochs: ' + str(num_of_epochs) + '\n')

    print(f'\t\t\tTrain\t\tValidation')
    print(f'\t--------------------------------------------')
    # Start the training loop
    for i in range(num_of_epochs):
        epoch_loss, val_epoch_loss = 0, 0
        num_of_steps_in_epoch, val_num_of_steps_in_epoch = 0, 0

        for step, (images_batch, axion_mass_batch) in enumerate(train_data_loader):
            optimizer.zero_grad()

            # Use GPU if available
            if torch.cuda.is_available():
                images_batch = images_batch.cuda()
                axion_mass_batch = axion_mass_batch.cuda()

            # RUN the model
            predicted_axion_mass = model(images_batch)
        
            # Calculate loss
            loss = mse_loss(predicted_axion_mass,axion_mass_batch)

            # Calculate gradient
            loss.backward()

            # Do an optimization step
            optimizer.step()

            epoch_loss+=loss
            num_of_steps_in_epoch+=1

        loss_w = round((epoch_loss/num_of_steps_in_epoch).detach().item(),5)

        with torch.no_grad():
            for step, (images_batch, axion_mass_batch) in enumerate(test_data_loader):
                # Use GPU if available
                if torch.cuda.is_available():
                    images_batch = images_batch.cuda()
                    axion_mass_batch = axion_mass_batch.cuda()

                # RUN the model
                predicted_axion_mass = model(images_batch)
            
                # Calculate loss
                val_loss = mse_loss(predicted_axion_mass,axion_mass_batch)

                val_epoch_loss+=val_loss
                val_num_of_steps_in_epoch+=1

        val_loss_w = round((val_epoch_loss/val_num_of_steps_in_epoch).detach().item(),5)

        if (i + 1) % 5 == 0:
            print(f'\tEpoch {i+1} loss:\t{loss_w}\t\t{val_loss_w}')

    print('\n\tDONE.\n')
    print('----------------------------------------------------')

    return model
