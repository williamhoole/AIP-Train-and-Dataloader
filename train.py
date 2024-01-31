import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim




from dataloader.dataset import MyDataset
from utils.util_functions import *
from models.torchmodels import get_model

def main(args):
    # # Define hyperparameters
    
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs


    train_dataset = MyDataset(args, training=True)
    test_dataset = MyDataset(args, training=False)

    # Split the dataset into training and testing sets
    train_data, _  = train_test_split(train_dataset, test_size=0.2, random_state=args.seed)
    _ , test_data = train_test_split(test_dataset, test_size=0.2, random_state=args.seed)

    # # Create the dataset and dataloader
    # dataset = MyDataset(args.data_dir, args.roi_info, training=True)
    # train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

    # # count the number of samples in each class
    train_class_counts = [0] * len(train_dataset.classes)
    for _, label in train_data:
        train_class_counts[label] += 1
    print("Train data class counts:", train_class_counts)

    test_class_counts = [0] * len(test_dataset.classes) 
    for _, label in test_data:
        test_class_counts[label] += 1
    print("Test data class counts:", test_class_counts)

    # # Create the samplers for the training and testing sets
    # train_class_counts = [0] * len(train_dataset.classes)
    # for _, label in train_data:
    #     train_class_counts[label] += 1
    # train_class_weights = [1.0 / count for count in train_class_counts]
    # train_weights = [train_class_weights[label] for _, label in train_data]
    # train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # test_class_counts = [0] * len(test_dataset.classes)
    # for _, label in test_data:
    #     test_class_counts[label] += 1
    # test_class_weights = [1.0 / count for count in test_class_counts]
    # test_weights = [test_class_weights[label] for _, label in test_data]
    # print(test_weights)
    # test_sampler = WeightedRandomSampler(test_weights, len(test_weights))

    # # Create the data loaders for the training and testing sets
    train_loader = DataLoader(train_data, batch_size=batch_size,)#sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=len(test_data),) #sampler=test_sampler)

    # # Create the model
    model = get_model(args.model, len(train_dataset.classes), pretrained=True)
    model = model.cuda()
    
    # # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # # Train the model
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs and labels
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 10))
                running_loss = 0.0
            
        # Evaluate the model on the test set
                
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Accuracy of the network on the test images: %d %%' % accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print('Saved best model')

        lr_scheduler.step()

    print('Finished Training')


def arguments():
    parser = argparse.ArgumentParser(description='Training and Augmentation Parameters')
    
    parser.add_argument('--data_dir', type=str, default='data', help='path to dataset')
    parser.add_argument('--roi_info', type=str, default='roi_info', help='path to roi_info')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--input_size', type=int, default=224, help='input size of image')
    parser.add_argument('--model', type=str, default='vgg16', help='model to train')
    parser.add_argument('--trainsplit', type=float, default=0.8, help='train split')

    parser.add_argument('--rotate', type=float, default=0.0, help='rotation probability')
    parser.add_argument('--flip', type=float, default=0.0, help='flip probability')
    parser.add_argument('--zoom', type=float, default=0.0, help='zoom probability')
    parser.add_argument('--shift', type=float, default=0.0, help='shift probability')
    parser.add_argument('--noise', type=float, default=0.0, help='noise probability')
    parser.add_argument('--squeeze', type=float, default=0.0, help='squeeze probability')
    
    

    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    main(args)  