### 此项目基于mmdetection模型：

https://github.com/open-mmlab/mmdetection

### 数据集增强手段基于DataAugmentation_ForObjectDetect：

https://github.com/DLLXW/DataAugmentation_ForObjectDetect

### 数据集标注：

labelImg

### 配置：

1、
打开mmdet/dataset/voc.py，把VOCDdataset中的classes改成自己类别

2、
打开mmdet/core/evaluation/class_name.py，把里面的voc_classes()改成自己数据集的类别

3、
打开mmdetection/configs/pacal_voc/faster_rcnn_r50_fpn_1x_voc0712.py把num_classes修改成自己数据集的类别数
4、
打开mmdetection/configs/base/models/faster_rcnn_r50_fpn.py,把下图的num_classes改成自己的类别数
5、
打开mmdetection/configs/base/datasets/voc0712.py,改成自己的路径

### 使用：

1、进入虚拟环境

2、进入文件夹下

3、创建WORK_DIR文件夹输入命令开始训练

python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py --work-dir WORK_DIR

--gpus 1

python tools/train.py WORK_DIR/faster_rcnn_r50_fpn_1x_voc0712.py 

4、测试

python demo/image_demo.py demo/0019.jpg WORK_DIR/faster_rcnn_r50_fpn_1x_voc0712.py WORK_DIR/epoch_200.pth

5、作图

python tools/analysis_tools/analyze_logs.py plot_curve WORK_DIR/20210321_141247.log.json --keys



mAP	loss_rpn_cls	loss	acc	loss_cls	loss_rpn_bbox

### WORK_DIR/epoch_200.pth：
测试结果：

![图片1](https://github.com/kxdongovo/Semiconductor-OCR/assets/55145471/073fb36e-5e09-4716-9c1b-2a61dee558a6)

mAP：

![图片2](https://github.com/kxdongovo/Semiconductor-OCR/assets/55145471/ed98b2a0-ec1a-45ba-8743-94197df9daa6)

acc：

![图片3](https://github.com/kxdongovo/Semiconductor-OCR/assets/55145471/ec38918d-2683-4bf9-b492-bdae73e65594)


