# NNDL_final
NNDL期末作业，小组成员为隋东言与李啸晗，课题为数据增强方法及性能探究。

训练LeNet，ALexNet，VGG19，GoogLeNet与WideResNet：
1.配置Python环境(Python3.6, Pytorch1.8.0, torchvision0.9.0)；
2.打开baseline，cutmix，cutout与mixup相关的.py文件，在load_model()中选择想要训练的模型(将其他模型添加注释)；
3.命令行中运行python xxxx.py即可。（tensorboard训练曲线位于./run文件夹中）

训练ResNet与NeuralOdeNet：
1.同上配置Python环境
2.在jupyter notebook中运行相应的.ipynb文件即可。

计算top1与top5准确率：
1.在jupyter notebook中打开top5.ipynb；
2.选择训练好的模型的.pth文件，将checkpoint改为文件名（cifar10和cifar100数据集要改相应的torchvision.datasets函数及函数输入的文件夹）然后运行即可。

CAM.ipynb用于绘制CAM图。
