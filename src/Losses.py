import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from render import Microfacet
device='cuda'
eps = 1e-6

def gyArray2PIL(im):
    return Image.fromarray((im*255).astype(np.uint8))

def gyApplyGamma(im, gamma):
    if gamma < 1: im = im.clip(min=eps)
    return im**gamma
def gyTensor2Array(im):
    return im.detach().cpu().numpy()

class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss 


def render_loss(img,texture,args):
    
    renderOBJ = Microfacet(res=args.res, size=args.image_size)
    result = torch.zeros(args.N,3,args.res,args.res).to(device)
    for i in range(args.N):
        result[i] = renderOBJ.eval(texture,args.light_pos[i],args.camera_pos[i],torch.from_numpy(args.light).cuda())
    
    criterion = torch.nn.MSELoss().to(device)
    result = result.clamp(eps,1)**(1/2.2)
    
    loss = criterion(img,result)
    
    loss_list = []
    loss_list.append(loss.item()*1000)
        
    return loss*1000 , loss_list
        
        
        
  
