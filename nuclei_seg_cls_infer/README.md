# Nuclei segmentation and classification Part

This folder is for the first step of our proposed framework.

This code is revised from the source code of [HoVerNet](https://github.com/vqdang/hover_net).

## Set Up Environment
Please set up your enviroment follow https://github.com/vqdang/hover_net

## Usage
Set up the file_path and out_dir in Demo.ipynb, and run it on jupter.

The predicted result of each ROIs will be save in ".mat" on the out_dir.

## MODEL File
The trained model (mircronet) need to be download from our nextcloud.

The link and passwd is shown in [download_link.txt](https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer/blob/master/nuclei_seg_cls_infer/micronet/download_link.txt)

Then mv the download model files (two files) to [micronet](https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer/tree/master/nuclei_seg_cls_infer/micronet)
