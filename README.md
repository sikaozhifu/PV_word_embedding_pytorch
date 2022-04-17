# PV_word_embedding_pytorch
多模态特征融合与词嵌入驱动的 三维模型分类检索方法



### 相关文件下载地址

数据集（datasets）：

pc：[pc](https://drive.google.com/file/d/1zfNS1VdoyVRh9z-cWQIeZQyqq-Hi6P9F/view?usp=sharing)

view：[view](https://drive.google.com/file/d/1ChyHuyEYEqMDlPVmm98UeU5lWaG-bV9J/view?usp=sharing)

模型（model）：

MVCNN-ALEXNET-save-ckpt.pth：[MVCNN-ALEXNET-save-ckpt.pth](https://drive.google.com/file/d/1rdvYVxQxsncwfeveLvE2KmKyc1Lt7xK6/view?usp=sharing)

GoogleNews-vectors-negative300.bin.gz：[GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/1fl5_6s5HmBEL4YPfHZGMQjnP10pUBq1U/view?usp=sharing)

PCNet-save-ckpt.pth：[PCNet-save-ckpt.pth](https://drive.google.com/file/d/1Dtn285uLCpBGaYc1a4j8pdUQvXEkajDt/view?usp=sharing)

PCNet-save-arg.pth：[PCNet-save-arg.pth](https://drive.google.com/file/d/1EGr-NEJqUhSaDjybYAFkNFr2ZVkAhpxk/view?usp=sharing)

PVNet2-ALEXNET.pth：[PVNet2-ALEXNET.pth](https://drive.google.com/file/d/1sdqSZWbNojRW4hpyJip5xQSsmFxxX51q/view?usp=sharing)



1. 将数据集下载到本地，并解压到data目录下面

2. 下载GoogleNews-vectors-negative300.bin.gz预训练模型到根目录

3. 下载预训练模型MVCNN-ALEXNET-save-ckpt.pth、PCNet-save-ckpt.pth、PCNet-save-arg.pth、PVNet2-ALEXNET.pth

4. 运行val.py文件，对模型进行评估。

5. 若要从头训练模型，则分别运行：

   > train_mvcnn.py
   >
   > train_pc.py
   >
   > train.py
