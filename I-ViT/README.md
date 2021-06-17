# Instance Based Vision Transformers

![ViT](./static/model.jpeg)


## Installation

Create the environment:

```bash
conda env create -f environment.yml
```

Preparing the dataset:

```bash
mkdir data
cd data
ln -s path/to/dataset imagenet
```

## Running the Scripts

 *prepare instance data*: 
 
 You should run the nuclei_seg_cls_infer project to get the nuclues segmentation and grading mask at first.
 
 Then you should set the crop_size (P=64), dataset_path, mask_path and output_dir before you run this program.
 
 The result of nuclues patch, grades and position will be saved in '.mat' on output_dir.
```bash
python crop_nuclues.py 
```



For *training*:

set the crop_path where you save the result of nuclues patch, grades and position.
```bash
python train_scnn_pos.py --crop_path=/home5/hby/PRCC/New_Data/crop
```

For *testing*:

set the crop_path where you save the result of nuclues patch, grades and position.
```bash
python scnn_pos_test.py  --crop_path=/home5/hby/PRCC/New_Data/crop
```


