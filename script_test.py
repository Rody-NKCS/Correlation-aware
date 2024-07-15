import sys
import os
import argparse
from src.optim import optim
import warnings
warnings.filterwarnings("ignore")




if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = "SVBRDF")
    parser.add_argument(
        '--path', 
        type = str,
        default = "data/",
        help = "Path of input images.",
    )
    parser.add_argument(
        '--out_path', 
        type = str,
        default = "outputs",
        help = "Output path of the results.",
    )
    parser.add_argument(
        '--epochs', 
        type = int, 
        default = 10, 
        help = "Iterations of the latent space optimization.",
    )
    
    parser.add_argument(
        '--sec_epochs',
        type = int,
        default = 500,
        help = "Iterations of fine-tuning of adapters.",
    )
    parser.add_argument(
        '--lr', 
        type = float, 
        default = 0.001,
    )
    parser.add_argument(
        '--num_render', 
        type = int,
        default = 9,
        help = "Number of renderings.",
        
    )
    parser.add_argument(
        '--N', 
        type = int,
        default = 4,
        help = "Number of input images.",
    )
    parser.add_argument(
        '--res', 
        type = int,
        default = 256,
        help = "The resolution of input image.",
    )
    parser.add_argument(
        '--seed', 
        type = int,
        default = 666,
    )
    args = parser.parse_args()
    
    optim(args)
    
    
    
    
    
    
    
    
    
    
    
    