import tensorflow as tf
from core.shuffle_net_v2 import shuffle_net_v2
import numpy as np
import cv2

Color = [[128, 0, 0],[0, 128, 0],[64, 128, 128],[192, 0, 0],[64, 128, 0],[192, 128, 128],
        [0, 64, 0],[128, 128, 0],[0, 0, 128],[128, 64, 0],[0, 192, 0],[128, 192, 0],[0, 64, 128],[128, 0, 128],[0, 128, 128],[128, 128, 128],
        [64, 0, 0],[192, 128, 0],[64, 0, 128],[192, 0, 128],[0, 0, 0]]
# 是否使用dpc结构5
DPC = True


def prediction():
    if DPC:
        file = './model/'
    else:
        file = './transmodel/'

    # 数据预处理，bgr、rgb通道转化，并将输入图像（1024, 2048, 3）的像素点取值从0-255缩小为[-1， 1]
    image = cv2.imread("test5.png")  # 读取一张图片，test5是cityscape数据集的测试集中第一张图
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    image0 = image[np.newaxis, :, :, :]
    image = (2.0 / 255.0) * tf.cast(image0, tf.float32) - 1.0

    # 输入网络中
    Shuffle = shuffle_net_v2(image, False)
    net, _ = Shuffle.shuffle_net()
    predicting = Shuffle.dense_predition_cell(net)
    predict = tf.argmax(Shuffle.prediction(predicting), 3)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(file)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 调用saver.restore()函数，加载训练好的网络模型
            print('Loading success')
        else:
            print('No checkpoint')
        pre_image = sess.run(predict)

        # 进行调试过程中输出一些数据
        net = sess.run([tf.get_collection("output"), tf.get_collection("input"),tf.get_collection("weight"), tf.get_collection("quant_weight"),tf.get_collection("bias"),tf.get_collection("quant_bias"), tf.get_collection("clip_w"), tf.get_collection("clip"), tf.get_collection("convTrue"), tf.get_collection("var"), tf.get_collection("mean"), tf.get_collection("gamma")])
        print("使用量化本层输出")
        print("type:", type(net[0][0]))
        print(np.shape(net[0][0]))
        print(np.ravel(net[0][0])[0: 20])
        nt1 = np.ravel(net[0][0])
        print("不使用量化本层输出")
        print("type:", type(net[8][0]))
        print(np.shape(net[8][0]))
        print(np.ravel(net[8][0])[0: 20])
        nt2 = np.ravel(net[8][0])
        nt = nt2 - nt1
        print("两者之间差距的最值:", np.max(nt), np.min(nt))
        nums = np.shape(np.where(np.abs(nt)>1.0))
        print('差距比1大的数量:', nums)


        print("输入")
        print(np.shape(net[1]))
        print(np.ravel(net[1])[0: 20])

        print("卷积核权重")
        print(np.shape(net[2][0]))
        print(np.ravel(net[2])[0: 100])
        nums = np.shape(np.where(np.abs(net[2][0]) > 8.0))
        print('weights >8:', nums)
        print(np.max(net[2]), np.min(net[2]))

        print("量化后卷积核权重")
        print(np.shape(net[3]))
        print(np.ravel(net[3])[0: 20])

        weightcha = np.ravel(net[3]) - np.ravel(net[2])
        print("量化前后卷积核差距：", weightcha[0:20])
        nums = np.shape(np.where(np.abs(weightcha)> 0.5))
        print('误差>0.5个数:', nums)
        print("误差极值", np.max(weightcha), np.min(weightcha))
        cha = net[3][0]-net[2][0]
        print(np.shape(cha))
        suoyin = np.where(np.abs(cha)>0.5)
        print("位置在：", suoyin)

        print("bias")
        print(np.shape(net[4]))
        print(np.ravel(net[4])[0: 20])
        print(np.max(net[4]), np.min(net[4]))

        print("quant_bias")
        print(np.shape(net[5]))
        print(np.ravel(net[5])[0: 20])
        print(np.max(net[5]), np.min(net[5]))

        print("clip_w=", net[6])
        print("clip=", net[7])

    # 绘图
    visual_anno = np.zeros((1024, 2048, 3), dtype=np.uint8)
    print(np.shape(pre_image))
    for i in range(1024):  # i for h
        for j in range(2048):
            color = Color[pre_image[0, i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]
    cv2.imshow("image", image0[0])
    cv2.imshow("img", visual_anno)
    cv2.waitKey(0)


prediction()


