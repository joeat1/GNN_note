# GNN

神经网络的迅速发展，也推动着将神经网络运用到图这一特殊的数据结构的相关研究，熟悉和运用图神经网络的方法，很有必要。
此处将收集和整理相关论文和笔记，便于入门，便于快速把握论文基本信息。

## 论文获取

+ 可以查询最新论文数据：[网址](http://arxitics.com/search?q=GNN&sort=updated)
+ 由Jie Zhou等整理的详细的论文列表： https://github.com/thunlp/GNNPapers 清华大学自然语言处理实验室
+ 知乎精华：https://www.zhihu.com/topic/20747184/top-answers

## 列表目录

### 综述

+ A Comprehensive Survey on Graph Neural Networks. arxiv 2019. [论文地址](https://arxiv.org/pdf/1901.00596.pdf)
  + 知乎笔记：https://zhuanlan.zhihu.com/p/54241746
+ Graph Neural Networks: A Review of Methods and Applications. arxiv 2018. [论文地址](https://arxiv.org/pdf/1812.08434.pdf)
  + CSDN笔记：https://blog.csdn.net/ssq183/article/details/101118929
+ Deep Learning on Graphs: A Survey. arxiv 2018. [论文地址](https://arxiv.org/pdf/1812.04202.pdf)
+ 图卷积神经网络综述 2020 [论文下载](https://kns.cnki.net/kcms/detail/11.1826.tp.20191104.1632.006.html)
  + 

### 模型

> [知乎：  图神经网络从入门到入门](https://zhuanlan.zhihu.com/p/136521625) 

#### GCN

> 基础知识 [参考](https://zhuanlan.zhihu.com/p/124727955) 
>
> + 特征值、特征向量与特征分解（谱分解）
> + 拉普拉斯矩阵
> + 傅里叶变化与卷积操作（特征抽取技术）

+ 谱域GCN  Spectral Networks and Locally Connected Networks on Graphs. ICLR 2014.  [论文地址](https://arxiv.org/pdf/1312.6203.pdf)

+ 空域GCN  Learning Convolutional Neural Networks for Graphs  ICML 2016.  [论文地址](https://proceedings.mlr.press/v48/niepert16.pdf)

  + 知乎笔记：https://zhuanlan.zhihu.com/p/27587371

+ 谱域GCN  Semi-Supervised Classification with Graph Convolutional Networks  ICLR 2017. [论文地址](https://arxiv.org/pdf/1609.02907.pdf)

  + 知乎笔记：https://zhuanlan.zhihu.com/p/31067515

+ **Inductive Representation Learning on Large Graphs.** NIPS 2017. [论文地址](https://arxiv.org/pdf/1706.02216.pdf) 

  + **Graph Sample and Aggregate(GraphSAGE)** 能够处理 large graphs，克服了GCN训练时内存和显存的限制，即使在图中加入新节点，也能计算出节点表示。

  + 训练时仅保留训练样本到训练样本的边（Inductive Learning），对邻居采用有放回的重采样/负采样方法进行定长抽样（Sample），之后汇聚（Aggregate）这些邻居的特征以更新自己信息。

  + 同时适用于有监督与无监督表示学习

  + 缺点：采样没有考虑到不同邻居节点的重要性不同，聚合时没有区分中心节点与邻居节点

#### GAT

+ **Graph Attention Networks.** ICLR 2018.  [论文地址](https://arxiv.org/pdf/1710.10903.pdf) 
  + 借鉴Transformer中self-attention机制，根据邻居节点的特征来分配不同的权值
  + 训练GCN无需了解整个图结构，只需知道每个节点的邻居节点即可
  + 为了提高模型的拟合能力，还引入了多头的self-attention机制



#### Graph Auto-Encoder(GAE)

+ Variational Graph Auto-Encoders（NIPS2016）  [论文地址](https://arxiv.org/pdf/1611.07308.pdf)
  + 知乎笔记：https://zhuanlan.zhihu.com/p/78340397
  + 将变分自编码器（Variational Auto-Encoders）迁移到图领域，用已知的图经过编码（图卷积）学到节点低维向量表示的分布（均值和方差），在分布中采样得到节点的向量表示，然后进行解码（链路预测）重新构建图。
  + 损失函数衡量生成图和真实图之间的差异，并加入各独立正态分布和标准正态分布的散度以限定各个正态分布的形式。



#### Graph Pooling

+ 简单的max pooling和mean pooling （不高效而且忽视了节点的顺序信息）
+ **Differentiable Pooling** (**DiffPool**)   Hierarchical Graph Representation Learning with Differentiable Pooling. NeurIPS 2018.  [论文下载](https://arxiv.org/pdf/1806.08804.pdf)
  + 通过一个**可微池化操作模块**去分层的聚合图节点



### 应用

+ Spam Review Detection with Graph Convolutional Networks  [论文地址](https://arxiv.org/pdf/1908.10679.pdf)
+ Abusive Language Detection with Graph Convolutional Networks [论文地址](https://arxiv.org/pdf/1904.04073.pdf)
+ **推荐系统**   Graph Convolutional Neural Networks for Web-Scale Recommender Systems   [论文地址](https://arxiv.org/pdf/1806.01973.pdf)
+ 

## 代码

> 大量基于tensorflow的代码都是1.x版本，如果安装的时2.x时，需要在代码中将`import tensorflow as tf 改为import tensorflow.compat.v1 as tf` 
>
> 代码基础知识：https://github.com/ageron/handson-ml 

+ PyG框架 Geometric Deep Learning Extension Library for PyTorch   https://github.com/rusty1s/pytorch_geometric
  + 文档：https://pytorch-geometric.readthedocs.io/
  + PyG包含有大量的基准数据集

+ Implementation of Graph Convolutional Networks in TensorFlow  https://github.com/tkipf/gcn  
  + 包含 multi-layer perceptron（多层感知机）gcn  ChebyshevGCN

## 数据集

+ https://github.com/shiruipan/graph_datasets  涉及化合物，引文网络，情感网络，脑功能等五个数据库
+ http://www.cs.umd.edu/~sen/lbc-proj/LBC.html 
  + 引文网络 CiteSeer、Cora、WebKB
  + 社交网络 Terrorists 
  + 参考 https://www.jianshu.com/p/67137451b67f
  + https://blog.csdn.net/yyl424525/article/details/100831452 
+ graph kernel datasets https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
+ Stanford Large Network Dataset Collection  https://snap.stanford.edu/data/index.html
+ https://github.com/awesomedata/awesome-public-datasets  
+ 图像数据集 http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm 
+ 视频数据集 https://research.google.com/youtube8m/download.html 
+ 情感文本数据集 http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/
+ 推荐系统数据集 https://research.yahoo.com/datasets   https://grouplens.org/datasets/hetrec-2011/

## 研究团队

> 此处列举的只是一部分，重要的学者远比我看到的多得多。

> **根据作者来搜索相关工作**
>
> 1. 个人网站（大部分学校所配置的，与其邮箱名有一定的关系），Github（部分编写了代码的论文一般会放在GitHub等平台上）
> 2. 数据库查询，利用Author项进行筛选（需要注意中文名与英文名存在的不同之处），并添加领域classification，机构/单位，和关键词abstract的限定，以增强查询的效果
> 3. 部分作者在某些数据库认证过，可以尝试在mendeley等管理软件打开后从details中，从网页查看论文，并获取作者的主页

+ Maosong Sun(孙茂松 教授), Zhiyuan Liu(刘知远 助理研究员), Jie Zhou, 清华大学计算机科学与技术学院
+ 沈华伟

## 其他资源

+ 图神经网络在视频理解中的探索 https://www.bilibili.com/video/av48201125/ 
+ 图神经网络介绍-Introduction to Graph Neural Network https://www.bilibili.com/video/av62661713
+ 唐杰-图神经网络及认知推理-图神经网络学习班  https://www.bilibili.com/video/av77934956
+ 图神经网络在线研讨会2020  https://www.bilibili.com/video/BV1zp4y117bB

# 说明

+ 本仓库仅作为学术研究使用，笔记不一定完全表述了原作者的想法，如果出现问题欢迎探讨，也建议与原作者联系。
+ 如果部分内容作者或出版社有异议，内容存在错误，请联系我进行删除或修改。

