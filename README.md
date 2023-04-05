 #数据增广（Data-Augmentation）
——————————————————————————————————————————————————————————————————————————————————————————————————————

1️⃣什么是"数据增广"？  
------------------------------------------------------------------------------------------------------------------------
  
当我们训练深度学习模型时，我们通常需要大量的数据来训练模型  

但是，有时候我们可能没有足够的数据来训练模型，或者我们的数据集可能存在一些问题，例如类别不平衡、过拟合等  

为了解决这些问题，我们可以使用数据增广技术。数据增广是指通过对原始数据进行一系列变换来生成新的数据，从而扩充原始数据集的大小  

这样可以增加模型的泛化能力，减少过拟合，并提高模型的性能  

在PyTorch中，我们可以使用transforms模块来实现数据增广  

transforms模块提供了很多常用的数据增广操作，例如随机水平翻转、随机旋转、随机缩放裁切等  

你可以将这些操作组合起来，生成一个数据增广操作序列，并将其应用到你的数据集上  

例如，在上面的代码中，我们使用了三个常用的数据增广操作：随机水平翻转、随机旋转和随机缩放裁切  

这些操作可以通过transforms.RandomHorizontalFlip、transforms.RandomRotation和transforms.RandomResizedCrop方法来实现  

——————————————————————————————————————————————————————————————————————————————————————————————————————

2️⃣四大数据增广方法  
-------------------------------------------------------------------------------------------------------------------------------

水平翻转、垂直翻转、随机旋转、随机裁切和随机色度变换被称为四大数据增广方法  

其中，水平翻转是将图像水平翻转180度，垂直翻转是将图像垂直翻转180度，随机旋转是将图像随机旋转一定角度，随机裁切是将图像随机裁剪一部分，而随机色度变换则是对图像的颜色进行随机变换 

这四种数据增强方法都是用来解决视觉问题的  

其中，水平翻转和垂直翻转可以解决平移不变性问题，随机旋转可以解决旋转不变性问题，随机裁切可以解决尺寸不变性问题，随机色度变换可以解决光照复杂性问题  

--------------------------------------------------------------------------------------------------------------------------------

在完成fork实验所需仓库后，通过PyCharm打开train.py的编辑，对代码改动后运行训练，训练结果如下图：  

![1](https://user-images.githubusercontent.com/128795948/229268531-c7fd8fe3-7334-43ff-b038-2f8a6b328a4e.PNG)

其中训练集与验证集各自保存在train文件夹与val文件夹中    

![2](https://user-images.githubusercontent.com/128795948/229268610-8f6e378f-cb80-464b-bee3-e77e2ea85e9a.PNG)  

![3](https://user-images.githubusercontent.com/128795948/229268731-6bc8cd1a-3607-4f45-bf08-2b4d48120f05.PNG)

之后通过bingAI，设计出通过模仿transforms，RandomResizedCrop实现数据增广的代码

    from torchvision import transforms
    import torch

    # 定义数据增广操作
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, p=0.5),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0), p=0.5)
    ])

    # 加载图像数据
    image = ... # 加载图像数据

    # 应用数据增广操作
    augmented_image = data_transforms(image)  

此代码使用 torchvision.transforms 模块中的 RandomHorizontalFlip，RandomRotation 和 RandomResizedCrop 函数来实现上述数据增广操作  

定义了一个数据增广操作序列，其中完成的增广包括：  

1、随机水平翻转（概率为50%）  

2、随机旋转（角度范围为±10度，概率为50%）  

3、随机缩放裁切（裁切后尺寸为256，缩放范围为0.8到1.0，概率为50%）

如果除了需要作用于训练集之外，还要作用于验证集，我们可以定义一个名为 augment_data 的函数  

该函数接受两个参数：image 和 is_training  

image 参数表示要增广的图像数据，is_training 参数表示当前是否处于训练阶段  

在该函数内部则如上图代码定义一个数据增广操作序列，来完成数据增广操作  

然后，我们应用这些数据增广操作并返回增广后的图像  

你可以在训练集和验证集上都调用这个函数来应用相同的数据增广操作  

以下是bingAI的示例代码：

    from torchvision import transforms
    import torch

    def augment_data(image, is_training):
        # 定义数据增广操作
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, p=0.5),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0), p=0.5),
            transforms.ToTensor()
        ])  

        # 应用数据增广操作
        augmented_image = data_transforms(image)
        return augmented_image

    # 加载图像数据
    image = ... # 加载图像数据  

    # 应用数据增广操作
    augmented_image = augment_data(image, is_training)  

——————————————————————————————————————————————————————————————————————————————————————————————————————

3️⃣利用PIL库尝试单张图片的数据增广  
------------------------------------------------------------------------------------------------------------

如果想对单个图像进行数据增强操作，则可以使用 `PIL` 库来加载图像，然后使用 `transforms` 来应用数据增强方法。  

我将该测试代码与测试图片都放入了use文件夹下，其中增广代码如下：  

    #读取原图
    import torchvision.transforms as transforms
    from PIL import Image
    img = Image.open('cat01.png')
    img.show()
    print('img:',img.size)  

    #比例缩放
    img1 = transforms.Resize(112)(img)  # 短边缩放成112，长边按比例缩放
    img2 = transforms.Resize((112, 112))(img)  # 强行缩放成手动设置的比例
    img1.show()
    img2.show()
    print('img1:', img1.size)
    print('img2:', img2.size)  

    #位置截取
    rand_img = transforms.RandomCrop(112)(img)  # 随机裁剪112*112
    rand_img.show()
    print('rand_img:', rand_img.size)
    center_img = transforms.CenterCrop(112)(img)  # 以原图中心为中心，裁剪112*112
    center_img.show()
    print("center_img:", center_img.size)  

    #翻转
    hor_img = transforms.RandomHorizontalFlip(p=1)(img)  # 随机水平翻转, p为概率
    hor_img.show()
    ver_img = transforms.RandomVerticalFlip(p=1)(img)  # 随机垂直翻转，p为概率
    ver_img.show()  

    #旋转
    rot_img = transforms.RandomRotation(15)(img)  # 随机在（-15， 15）度旋转
    rot_img.show()
    print("rot_img:", rot_img.size)  

    #亮度、对比度和色调的变化
    bright_img = transforms.ColorJitter(brightness=1)(img)  # 随机从0~2之间的亮度变化
    bright_img.show()
    contrast_img = transforms.ColorJitter(contrast=1)(img)  # 随机从0~2之间的对比度
    contrast_img.show()
    hue_img = transforms.ColorJitter(hue=0.5)(img)  # 随机从-0.5~0.5之间的色调
    hue_img.show()
    saturation_img = transforms.ColorJitter(saturation=0.5)(img)
    saturation_img.show()
    color_img = transforms.ColorJitter(brightness=1,contrast=1,hue=0.5,saturation=0.5)(img)
    color_img.show()  
  
以下为cat.png经过增广后的效果图：  
![4](https://user-images.githubusercontent.com/128795948/229270421-1317264b-203c-4bf8-8f2d-b2cb4b9b947b.PNG)

![5](https://user-images.githubusercontent.com/128795948/229270426-893978d5-fe2c-491b-9b94-1ea864974a47.PNG)

![6](https://user-images.githubusercontent.com/128795948/229270429-ff237293-c625-4733-b0e1-dccb48719d4f.PNG)

但是该代码并没有能将完成增广后的数据直接导入特定文件夹的部分

——————————————————————————————————————————————————————————————————————————————————————————————————————
