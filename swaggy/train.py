import torch

def mse_loss(pred, true):
    loss = (pred-true).pow(2)
    return loss.mean()

def train(model,optimizer,train_data_loader,test_data_loader,scheduler=None,num_of_epochs=1):
    print('----------------------------------------------------')
    print('Begin Non-SWAG training')
    print('----------------------------------------------------\n')
    print('\tTraining set-up:')
    print('\t----------------\n')
    print('\tNumber of epochs: ' + str(num_of_epochs) + '\n')
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

        loss_w = (epoch_loss/num_of_steps_in_epoch).detach().item()

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

        val_loss_w = (val_epoch_loss/val_num_of_steps_in_epoch).detach().item()

        print(f'\t\tEpoch {i+1} loss:\n\tTrain: {loss_w}\n\tVal: {val_loss}')

    print('\n\tDONE.\n')
    print('----------------------------------------------------')

    return model
