import torch
import torch.nn as nn
import os
import sys
import numpy as np
from PIL import Image
from src.util import *
import random
sys.path.append("src/") 
from models.network import Encoder_GCN, Encoder 
from models.network import Decoder
from src.Losses import render_loss
from render import tex2png

from torch.optim import Adam
from tqdm import tqdm

device = 'cuda'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     
def loadLightAndCamera(in_dir):
    print('Load camera position from ', os.path.join(in_dir, 'camera_pos.txt'))
    camera_pos = np.loadtxt(os.path.join(in_dir, 'camera_pos.txt'), delimiter=',').astype(np.float32)

    print('Load light position from ', os.path.join(in_dir, 'light_pos.txt'))
    light_pos = np.loadtxt(os.path.join(in_dir, 'light_pos.txt'), delimiter=',').astype(np.float32)

    im_size = np.loadtxt(os.path.join(in_dir, 'image_size.txt'), delimiter=',')
    im_size = float(im_size)
    light = np.loadtxt(os.path.join(in_dir, 'light_power.txt'), delimiter=',')

    return light_pos, camera_pos, im_size, light
        

def optim(args):
    setup_seed(args.seed)
    img_channel = 3
    width = 64
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
   
    criterion = nn.MSELoss().to(device)
    args.out_path = args.out_path + str(args.N)
    
    for name in os.listdir(args.path):
        print(name)
        if args.N >= 4:
            enc = Encoder_GCN(img_channel=img_channel, width=width, enc_blk_nums=enc_blks).to(device)
            enc.load_state_dict( {k.replace('module.',''):v for k,v in torch.load('ckpt/enc_4.ckpt',map_location='cuda:0').items()})
            dec = Decoder(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)
            dec.load_state_dict({k.replace('module.',''):v for k,v in torch.load('ckpt/dec_4.ckpt',map_location='cuda:0').items()})
            
        else:
            enc = Encoder(img_channel=img_channel, width=width, enc_blk_nums=enc_blks).to(device)
            enc.load_state_dict( {k.replace('module.',''):v for k,v in torch.load('ckpt/enc_1.ckpt',map_location='cuda:0').items()})
            dec = Decoder(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)
            dec.load_state_dict({k.replace('module.',''):v for k,v in torch.load('ckpt/dec_1.ckpt',map_location='cuda:0').items()})
            args.lr = 0.0001
      
        opt_path = os.path.join(args.path,name)
        args.light_pos, args.camera_pos, args.image_size, args.light = loadLightAndCamera(opt_path)
        out_opt_path = os.path.join(args.out_path,name)
        
        if not os.path.exists(out_opt_path):
            os.makedirs(out_opt_path)
        
        img = []
        
        for i in range(args.N):
            im = Image.open(os.path.join(opt_path,str(i)+'.png'))
            im = gyPIL2Array(im)
            img.append(im)
        img = np.asarray(img)
        img = torch.from_numpy(img)
        img = img.permute(0,3,1,2).to(device)
        args.res = img.shape[-1]
        
        with torch.no_grad():
            if args.N >= 4:
                x = enc(img[0].unsqueeze(0),img[1].unsqueeze(0),img[2].unsqueeze(0),img[3].unsqueeze(0))
            else:
                x = enc(img[0].unsqueeze(0))
      
        x[-1].requires_grad = True
        optimizer = Adam([x[-1]], lr=args.lr, betas=(0.9, 0.999))
        
        for epoch in tqdm(range(args.epochs)):
            texture = dec(x)
            loss,loss_list = render_loss(img,texture,args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        x[-1].requires_grad = False
        optimizer = torch.optim.Adam([
    	 {'params': dec.diff_conv.parameters(), 'lr': args.lr}, 
    	 {'params': dec.normal_conv.parameters(), 'lr': args.lr},
         {'params': dec.rough_conv.parameters(),  'lr': args.lr},
         {'params': dec.spec_conv.parameters(), 'lr':args.lr}
    	 ])
        
        for epoch in tqdm(range(args.sec_epochs)):
            texture = dec(x)
            loss,loss_list = render_loss(img,texture,args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        texture = dec(x)
        fn = os.path.join(out_opt_path,'tex.png')
        png = tex2png(texture, fn)
        


    
            
        
        
        
      
