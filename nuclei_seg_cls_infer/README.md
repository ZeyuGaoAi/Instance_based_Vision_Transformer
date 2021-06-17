# Nuclei segmentation and classification Part

This folder is for the first step of our proposed framework.

This code is revised from the source code of [HoVerNet](https://github.com/vqdang/hover_net/tree/tensorflow-final).

## Set Up Environment
Please set up your enviroment follow https://github.com/vqdang/hover_net/tree/tensorflow-final

## Usage
Set up the file_path and out_dir in Demo.ipynb, and run it on jupter.

The predicted result of each ROIs will be save in ".mat" on the out_dir.

## Moder File
The trained model (mircronet) need to be download from our nextcloud.

The link and passwd is shown in [download_link.txt](https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer/blob/master/nuclei_seg_cls_infer/micronet/download_link.txt)

Then mv the download model files (two files) to [micronet](https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer/tree/master/nuclei_seg_cls_infer/micronet)

## Dataset
The dataset to train the nuclei segmentation and classification (grading) model can be download from [This Link](https://dataset.chenli.group/home/ccrcc-grading)

This dataset is original used for ccRCC grading, but part of the ROIs are selected from pRCC WSIs, cause these two subtypes have a same grading guide line.
