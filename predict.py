# Imports here
import json
from torchvision import models
import torch 
import numpy as np
import argparse 
import copy
from PIL import Image


# create argument parser
parser = argparse.ArgumentParser()

#Add arguments to parser
parser.add_argument("--img_path", type=str, default=" ", help="Path for image you wanna predict")
parser.add_argument("--class_names", type=str, default="cat_to_name.json", help="Path for json file with class names")
parser.add_argument("--model", type=str, default="./checkpoint.pth", help="Path for pretrained model to predict ")
parser.add_argument("--topk", type=int, default=5, help="Set number of top classes you want to see defult is 5")
parser.add_argument("--gpu", type=bool, default=False, help="use gpu while predict or not ")

# Parse the arguments 
args = parser.parse_args()

img_path = args.img_path
cat_to_name = args.class_names
checkpoint = args.model
topk = args.topk
gpu = args.gpu 

#Imports cat_to_name.json file
with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f, strict=False)

#set device
# if gpu:
#     device = 'cuda'
# else:
#     device = 'cpu'

device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu' )

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(chekpoint_file):
    
    checkpoint = torch.load(chekpoint_file)

    if checkpoint['arch'] == 'vgg16':
         model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
         model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #  Process a PIL image for use in a PyTorch model
    # Load the image
    image = Image.open(image_path)

    # Resize the image
    image.thumbnail((256, 256))

    # Crop the image
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))

    # Convert the image to a Numpy array and normalize
    np_image = np.array(image) / 255.0

    # Normalize the image
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    # Reorder the dimensions
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Load the model checkpoint
    model_checkpoint = load_model(chekpoint_file=model)
    
    # Move the image tensor to the same device as the model
    model_checkpoint = model_checkpoint.to(device)
    
    # Load and preprocess the image
    image = process_image(image_path)
    image = np.expand_dims(image, axis=0)
    img_tensor = torch.from_numpy(image).type(torch.FloatTensor).to(device)
    
    # Set the model to evaluation mode
    model_checkpoint.eval()
    
    # Pass the image tensor through the model
    with torch.no_grad():
        output = model_checkpoint.forward(img_tensor)
    
    # Calculate the probabilities
    probabilities = torch.exp(output)
    
    # Get the top k probabilities and their corresponding indices
    top_probs, top_indices = torch.topk(probabilities, topk)
    
    # Convert the indices to class labels
    idx_to_class = {v: k for k, v in model_checkpoint.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    # Convert the probabilities and class labels to lists
    top_probs = top_probs[0].tolist()
    top_classes = [str(c) for c in top_classes]
    
    return top_probs, top_classes

prob, classs = predict(img_path,checkpoint,topk )

flower_names = [cat_to_name[str(i)] for i in classs]

print(flower_names)
print(prob)
