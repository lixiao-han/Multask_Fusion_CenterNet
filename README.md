# Multask_Fusion_CenterNet
多任务特征融合的CenterNet模型，专用于检测动态目标

最好/最合适的网络结构为pose_dla_dcn70.0，位于：训练CenterNet尝试的所有网络们/包含注意力（分割模块）/CA注意力模块/pose_dla_dcn(70.0_alter43.1'sMaxpoolToDownbilinear).py，训练详情可在：Multask_Fusion_CenterNet/object detection/src/runs70.0_alter43.1'sMaxpoolToDownbilinear中查看
具体查看方法：
cd Multask_Fusion_CenterNet/object detection/src
#运行tensorboard查看训练详情
tensorboard --logid=runs70.0_alter43.1'sMaxpoolToDownbilinear

#训练方法：
python main.py ctdet --val_intervals 5 --exp_id coco_dla_1x2.0 --dataset uadetrac1on10_b --arch dlav0 --batch_size 4 --master_batch 4 --lr 5e-4 --gpu 0

训练UA-DETRAC结果可视化：
![image](https://github.com/lixiao-han/Multask_Fusion_CenterNet/assets/82953938/d351bcfa-525e-4872-9853-8329195c7888)

由于UA-DETRAC数据集本身存在很多被忽略（无标记）区域，多多少少会影响检测性能，因此本方法通过mask_ignore.py方法对测试集图片存在的ignore区域进行了mask，mask后的效果如下图
![image](https://github.com/lixiao-han/Multask_Fusion_CenterNet/assets/82953938/aa5fb141-5dd5-4013-988e-48bb4232510f)
（该脚本文件对所有视频的ignore区域都进行了标注，也可通过更改mask_ignore.py中的代码自行mask）


