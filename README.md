# Correlation-aware Encoder-Decoder with Adapters for SVBRDF Acquisition
This is the code of "Correlation-aware Encoder-Decoder with Adapters for SVBRDF Acquisition" (Proceedings of SIGGRAPH Asia 2024). [Project](https://rody-nkcs.github.io/SVBRDF/) | [Paper]().
<img src='teaser.png'>

## Dependencies
- Python (with pillow, numpy; tested on Python 3.9)
- Pytorch (tested on 2.1.1 +  CUDA 12.4)

## Inference
Before running, please download:
1. Our pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1BebohTKZfpVQ6bPiT7AYh930pBAnZSWW?usp=sharing). Download them and extract them into `./ckpt/`.
2. Dataset. The input images can be rendered from [synthetic dataset](https://github.com/valentin-deschaintre/Single-Image-SVBRDF-Capture-rendering-loss) or [captured](https://github.com/tflsguoyu/svbrdf-diff-renderer).
   The input images should be organized in the following format:
   ```

   ```

### Inference on multi-image
Please use this command:
```
python script_test.py
  --N 4 # Number of input images
  --path "data/"  # Path of input images
  --out_path "outputs4" # Output path of the results
  --epochs 10 # Iterations of the latent space optimization
  --sec_epochs 500 # Iterations of fine-tuning of adapters
```
## Inference on single image
Please use this command:
```
python script_test.py
  --N 1 # Number of input images
  --path "data/"  # Path of input images
  --out_path "outputs1" # Output path of the results
  --epochs 10 # Iterations of the latent space optimization
  --sec_epochs 500 # Iterations of fine-tuning of adapters
```

## Citation

If you find this work useful for your research, please cite:



## Contact

This code is not clean version, will clean it up soon. feel free to email me if you have any questions: Rody1911641@gmail.com. Thanks for your understanding!
