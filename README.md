# Spatial information inference net: Road extraction using road-specific contextual information

Code for the paper:"Spatial information inference net: Road extraction using road-specific contextual information" by Chao Tao, Ji Qi, Yansheng Li, Hao Wang and Haifeng Li.

![img](fig\fig_03.png)

## Dependencies
```
python3
pytorch >= 1.1
```

## Datasets
### The CVPR dataset
The DEEPGLOBE-CVPR 2018 road extraction sub-challenge dataset (CVPR dataset) can be download from the competition [website](https://competitions.codalab.org/competitions/18467). Thanks!

### The Massachusetts road dataset
The Massachusetts road dataset was presented by Mnih and Hinton et al..
We collected the dataset from their [website](https://www.cs.toronto.edu/vmnih/data). Thanks!

### The RoadTracer dataset
The RoadTracer dataset was presented by Bastani et al. in *RoadTracer: Automatic Extraction of Road Networks from Aerial Images*. We collected this dataset and   conduct comparison experiments by using the [code](https://github.com/mitroadmaps/roadtracer) provided by the author. Thanks!

Images and GT_labels for training and testing after cropping shoulded be organized as follows:
```
├── root
|   ├── train
|   ├── train_labels
|   ├── val
|   ├── val_labels
```
In addition, the values of GT_labels are 0 and 1 (0-bg, 1-Road).


## Running
Once the data set is prepared, set the dir of your dataset at `config\opt_cvpr.py`, `config\opt_rbdd.py` and `config\opt_mit.py`. Then, training with `main_road_train.py`.

Do prediecting (or evaluating) use `main_road_reval.py` for the CVPR and Massachusetts dataset;
Do prediecting (or evaluating) use `main_road_reval_mit.py` and `eval_roadtracer.py` for the RoadTracer dataset;


## Citing this work
[1]	C. Tao, J. Qi, Y. Li, H. Wang, and H. Li, "Spatial information inference net: Road extraction using road-specific contextual information," ISPRS-J. Photogramm. Remote Sens., vol. 158, pp. 155-166, 2019.

@article{siinet2019,
   author = {Tao, Chao and Qi, Ji and Li, Yansheng and Wang, Hao and Li, Haifeng},
   title = {Spatial information inference net: Road extraction using road-specific contextual information},
   journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
   volume = {158},
   pages = {155-166},
   ISSN = {0924-2716},
   year = {2019},
   type = {Journal Article}
}
