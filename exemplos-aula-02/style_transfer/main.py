import torch
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from loss import get_style_model_and_losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

plt.ion()

IMGSIZE = 256

loader = transforms.Compose([
    transforms.Resize(IMGSIZE),  
    transforms.ToTensor()])  

def open_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)      
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 


def run_style_transfer(cnn, 
                       content_img, 
                       style_img, 
                       input_img, 
                       num_steps=300,
                       style_weight=1000000, 
                       content_weight=1):
    
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img,
        ['conv_4'],
        ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'])

    
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)
            
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == "__main__":

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    
    style_img = open_image("picasso.jpg")
    content_img = open_image("dancing.jpg")

    input_img = content_img.clone()

    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')
    
    output = run_style_transfer(cnn, content_img, style_img, input_img)

    plt.figure()
    imshow(output, title='Output Image')

    plt.ioff()
    plt.show()
