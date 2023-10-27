# Imports here
import json
from torchvision import transforms, models, datasets
import torch 
from torch import optim
import torch.nn as nn
import time
import argparse 
import copy
from collections import OrderedDict


#Imports cat_to_name.json file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# create argument parser
parser = argparse.ArgumentParser()

#Add arguments to parser
#UPDATE:Converting an optional argument to a positional argument data-dir  
# parser.add_argument("--data_dir", type=str, default="./flower_data", help="Set dataset directory to train")
parser.add_argument("data_dir", type=str, help="Set dataset directory to train")

parser.add_argument("--save_dir", type=str, default="./checkpoint.pth", help="Set directory to save model checkpoint")
parser.add_argument("--arch", type=str, default="vgg16", help="Choose Model architecture whether vgg16 or densenet121 ")
#hyperparameters
parser.add_argument("--learning_rate", type=float, default=0.001, help="Choose learning rate")
parser.add_argument("--hidden", type=int, default=1024, help="Set number of hidden layers")
parser.add_argument("--epochs", type=int, default=5, help="Set number of epochs to train model")
parser.add_argument("--gpu", type=bool, default=False, help="use gpu while train or not ")
parser.add_argument('--dropout', type=float, default=0.5, help='Determines probability rate for dropouts')

# Parse the arguments 
args = parser.parse_args()
# Access the argument values 
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_layer = args.hidden
epochs = args.epochs
gpu = args.gpu
dropout = args.dropout

#set device
# if gpu:
#     device = 'cuda'
# else:
#     device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu' )


# set dataset dirs
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
}
# Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform =data_transforms['train']),
    'validation' : datasets.ImageFolder(valid_dir, transform =data_transforms['validation']),
    'test': datasets.ImageFolder(test_dir, transform =data_transforms['validation'])
}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'trainLoader': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=4),
    'validLoader': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, num_workers=4),
    'testLoader': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, num_workers=4),

}

# Define Classifier
def classifier(arch='vgg16', dropout=0.5, hidden_layer=1024):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    
    for param in model.parameters():
        param.requires_grad = False

    my_classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_layer)),
            ('relu', nn.ReLU()),
            ('Dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_layer, 256)),
            ('output', nn.Linear(256, 102)),
            ('softmax', nn.LogSoftmax(dim = 1))]))

    model.classifier = my_classifier

    return model

#define model , criterion, optimizer
model = classifier(arch, dropout, hidden_layer)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# train model function
def train(model, criterion, optimizer, epochs, device):

    model.to(device)    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    steps = 0
    printe = 20
    acc = 0
    train_losses, valid_losses = [], []
    
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}.. ")
        print("-"*10)
        running_loss = 0 
        for images , labels in dataloaders['trainLoader']:
            steps +=1
            # pass images and labels to cuda
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # forward track train loss 
            #UPDATE: model.forward outdated the new way is model() for forward
            model_out = model.forward(images)
            loss = criterion(model_out, labels)

            #backward + optimize
            loss.backward()
            optimizer.step()
            
            # train statistics
            running_loss += loss.item() * images.size(0)

            # set model to evaluation mode to evaluate
            if steps % printe ==0 :
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for image, label in dataloaders['validLoader']:
                        image, label = image.to(device), label.to(device)
                        valid_out = model.forward(image)
                        vloss = criterion(valid_out, label)

                        valid_loss += vloss.item()
                         # Calculate accuracy
                        ps = torch.exp(valid_out)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == label.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                train_losses.append(running_loss/len(dataloaders['trainLoader']))
                valid_losses.append(valid_loss/len(dataloaders['validLoader']))
                running_loss = 0
                acc = (accuracy/len(dataloaders['validLoader']))*100
                print(f"Train loss: {train_losses[-1]:.3f}.. "
                              f"valid loss: {valid_losses[-1]:.3f}.. "
                              f"valid accuracy: {acc:.3f}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                model.train()
            
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model    

# Do validation on the test set
def test_network(model):
    no_correct =0
    total = 0
    model.to(device)
    with torch.no_grad():
        for images, labels in dataloaders['testLoader']:
            images, labels = images.to(device), labels.to(device)
            
            test_out = model(images)
            _, predicted = torch.max(test_out.data, 1)
            total += labels.size(0)
            no_correct += (predicted == labels).sum().item()
            
    return 100 * no_correct / total


train_model = train(model, criterion, optimizer, epochs, device)
test_accuracy = test_network(train_model)
print('Accuracy of the network on test images: %d %%'% test_accuracy)

#  Save the checkpoint 
train_model.class_to_idx = image_datasets['train'].class_to_idx

def checkpoint(state, save_dir):
    torch.save(state, save_dir)

checkpoint({
    'arch': arch,
    'classifier': train_model.classifier,
    'optimizer': optimizer.state_dict(),
    'state_dict': train_model.state_dict(),
    'class_to_idx': train_model.class_to_idx,
}, save_dir)