# GNN

神经网络的迅速发展，也推动着将神经网络运用到图这一特殊的数据结构的相关研究。

图是一种**非欧式**结构的结构化数据，它由一系列的对象（nodes）和关系类型（edges）组成，具有局部连接的特点，能表示更为复杂的信息；熟悉和运用图神经网络的方法很有必要。

基于此，将收集和整理相关论文和笔记，便于入门和个人回顾，便于快速发现论文基本信息和主要资源。



## 论文获取

+ 在[网址](http://arxitics.com/search?q=GNN&sort=updated)中查询最新论文数据；
+ 由Jie Zhou等整理的详细的[论文列表](https://github.com/thunlp/GNNPapers) ；
+ 知乎相关[话题精华](https://www.zhihu.com/topic/20747184/top-answers)



+ 部分相关关键词：关系推理，关系预测，知识图谱；图表示学习，图嵌入；节点分类



## 列表目录

### 综述

> 需要注意的是，几篇综述的参数表示和图GNN的分类角度存在不同，不要搞混

+ A Comprehensive Survey on Graph Neural Networks. arxiv 2019. [论文地址](https://arxiv.org/pdf/1901.00596.pdf)
  + [知乎笔记](https://zhuanlan.zhihu.com/p/54241746) 
  + 作者将图神经网络分为四类:循环图神经网络、卷积图神经网络、图自动编码器和时空图神经网络；并总结了图神经网络的数据集、开放源代码和模型评估。

+ Graph Neural Networks: A Review of Methods and Applications. arxiv 2018. [论文地址](https://arxiv.org/pdf/1812.08434.pdf)
  + [CSDN笔记](https://blog.csdn.net/ssq183/article/details/101118929)
  + 作者将GNN划分为五大类别，分别是：图卷积网络（Graph Convolution Networks，GCN）、 图注意力网络（Graph Attention Networks）、图自编码器（ Graph Autoencoders）、图生成网络（ Graph Generative Networks） 和图时空网络（Graph Spatial-temporal Networks）

+ Deep Learning on Graphs: A Survey. arxiv 2018. [论文地址](https://arxiv.org/pdf/1812.04202.pdf) 

+ 图卷积神经网络综述 2020 [论文下载](https://kns.cnki.net/kcms/detail/11.1826.tp.20191104.1632.006.html)


### 模型

> [知乎：  图神经网络从入门到入门](https://zhuanlan.zhihu.com/p/136521625) 

#### 图卷积网络（Graph Convolution Networks，GCN）

> GCN的本质目的就是用来提取拓扑图的空间特征。基于频谱的方法建立在**全局的**归一化的图拉普拉斯矩阵（实对称矩阵，是**无向图**的数学表示）之上，故而假定了图为无向图，难以处理大规模图，有向图以及动态图；而基于空间的方法较为灵活，可以引入采样技术来提高效率，从而近年来基于空间的模型越来越受到关注。

**基于频谱** spectral domain

> 基础知识可以 [参考](https://zhuanlan.zhihu.com/p/124727955) 
>
> + 特征值、特征向量与[特征分解](https://baike.baidu.com/item/%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3)（谱分解）
> + 拉普拉斯算子（描述的是**二维空间点与上、下、右邻居局部差异的和**）与[**拉普拉斯矩阵**](https://en.wikipedia.org/wiki/Laplacian_matrix)（对角线元素对应着各个结点的度数，非对角线元素对应着图的邻接矩阵）
> + 傅里叶变化与卷积操作（特征抽取技术）

+ Spectral Networks and Locally Connected Networks on Graphs. ICLR 2014.  [论文地址](https://arxiv.org/pdf/1312.6203.pdf)
+ Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. NIPS 2016. [论文地址](https://arxiv.org/pdf/1606.09375.pdf) 
+ Semi-Supervised Classification with Graph Convolutional Networks  ICLR 2017. [论文地址](https://arxiv.org/pdf/1609.02907.pdf)
  + [知乎笔记](https://zhuanlan.zhihu.com/p/31067515) 

**基于空域** spatial domain

> 每层 GCN 网络就是对邻居结点的特征进行聚合的操作，但提取特征的效果可能没有频域卷积好。
>
> 主要问题：如何获取邻居节点（采样），如何处理邻居节点的特征（聚合）

+ 空域GCN  Learning Convolutional Neural Networks for Graphs  ICML 2016.  [论文地址](https://proceedings.mlr.press/v48/niepert16.pdf)
  + [知乎笔记](https://zhuanlan.zhihu.com/p/27587371)
+ GraphSAGE：**Inductive Representation Learning on Large Graphs.** NIPS 2017. [论文地址](https://arxiv.org/pdf/1706.02216.pdf) 
  + **Graph Sample and Aggregate(GraphSAGE)** 能够处理 large graphs，克服了GCN训练时内存和显存的限制，即使在图中加入新节点，也能计算出节点表示。
  + 训练时仅保留训练样本到训练样本的边（Inductive Learning），对邻居采用有放回的重采样/负采样方法进行定长抽样（Sample），之后汇聚（Aggregate）这些邻居的特征以更新自己信息。
  + 同时适用于有监督与无监督表示学习
  + 缺点：采样没有考虑到不同邻居节点的重要性不同，聚合时没有区分中心节点与邻居节点



#### 图注意力网络（Graph Attention Networks）GAT

> 注意力机制能够放大数据中最重要的部分产生的影响。可以利用注意力函数，自适应地控制相邻节点j对节点i的贡献，或者集成多个模型，或者用于引导节点采样。

+ **Graph Attention Networks.** ICLR 2018.  [论文地址](https://arxiv.org/pdf/1710.10903.pdf) 
  + 借鉴Transformer中self-attention机制，根据邻居节点的特征来分配不同的权值
  + 训练GCN无需了解整个图结构，只需知道每个节点的邻居节点即可
  + 为了提高模型的拟合能力，还引入了多头的self-attention机制



#### 图自编码器（Graph Auto-Encoder）(GAE)

> 其目的是利用神经网络结构将图的顶点表示为低维向量。

+ Variational Graph Auto-Encoders（NIPS2016）  [论文地址](https://arxiv.org/pdf/1611.07308.pdf)
  + [知乎笔记](https://zhuanlan.zhihu.com/p/78340397)
  + 将变分自编码器（Variational Auto-Encoders）迁移到图领域，用已知的图经过编码（图卷积）学到节点低维向量表示的分布（均值和方差），在分布中采样得到节点的向量表示，然后进行解码（链路预测）重新构建图。
  + 损失函数衡量生成图和真实图之间的差异，并加入各独立正态分布和标准正态分布的散度以限定各个正态分布的形式。


#### Graph Pooling

+ 简单的max pooling和mean pooling （不高效而且忽视了节点的顺序信息）
+ **Differentiable Pooling** (**DiffPool**)   Hierarchical Graph Representation Learning with Differentiable Pooling. NeurIPS 2018.  [论文下载](https://arxiv.org/pdf/1806.08804.pdf)
  + 通过一个**可微池化操作模块**去分层的聚合图节点


### 应用

> 图神经网络广泛用于计算机视觉、推荐系统、交通拥堵情况、生物化学结构、社交网络信息等

+ Spam Review Detection with Graph Convolutional Networks  [论文地址](https://arxiv.org/pdf/1908.10679.pdf)
+ Abusive Language Detection with Graph Convolutional Networks [论文地址](https://arxiv.org/pdf/1904.04073.pdf)
+ **推荐系统**   Graph Convolutional Neural Networks for Web-Scale Recommender Systems   [论文地址](https://arxiv.org/pdf/1806.01973.pdf)



## 挑战与疑惑

+ 浅层结构 VS 深层网络
  + 目前的GNN还只能在较浅层的网络上发挥优势，而深层GCN性能会出现下降；
  + 社交网络中存在“六度空间”理论，多层卷积将使得节点获取了所有节点的信息而失去了局部性，并且可能使得所有节点的表示趋于相同，即过于平滑
+ 可拓展性 VS 完整性
  + 大规模的图网络计算量大难以处理，通过采样和聚类等方法都会丢失部分图的信息；
+ 同构图 VS 异构图
  + 异构图中包含了大量类型的节点和边的信息，同构图的方法未能良好运用
  + 没有较为合理的通用方法解决异构图数据信息
+ 动态图 VS 静态图
  + 目前大多方法只能应用在静态图上，对于动态图还没有特别好的解决方案



## 代码

> 大量基于tensorflow的代码都是1.x版本，如果安装的时2.x时，需要在代码中将`import tensorflow as tf 改为import tensorflow.compat.v1 as tf` 
>
> [部分基础知识](https://github.com/ageron/handson-ml )

+ [PyG框架](https://github.com/rusty1s/pytorch_geometric) Geometric Deep Learning Extension Library for PyTorch   
  + [官方文档](https://pytorch-geometric.readthedocs.io/)
  + PyG包含有大量的基准数据集

+ Implementation of Graph Convolutional Networks in TensorFlow  https://github.com/tkipf/gcn  
  + 包含 multi-layer perceptron（多层感知机）gcn  ChebyshevGCN

## 数据集

> 部分常用的数据集和说明已经放在datasets文件夹下

+ **引文网络** 
  + Cora
    + .content文件包含论文描述，每一行：<paper_id> <word_attributes>+ <class_label> ，特征应该有 1433 个维度
    + .cites文件包含语料库的引用关系，每一行：< ID of cited paper >   < ID of citing paper>
    + 处理方式 [参考](https://github.com/tkipf/gcn/blob/master/gcn/utils.py)
  + CiteSeer
    + 数据集包含3312种科学出版物，分为六类之一。引用网络由4732个链接组成。数据集中的每个出版物都用一个0/1值的词向量描述，该词向量指示字典中是否存在相应的词。该词典包含3703个独特的单词。
+ **社交网络**
  + BlogCatalog数据集是一个社交网络，它由博客作者及其社交关系组成。博客的类别代表了他们的个人兴趣。
  + Reddit数据集是一个无向图，由从Reddit论坛收集的帖子组成。如果两个帖子包含同一用户的评论，就会被链接。每个帖子都有一个标签，表明它所属的社区。
+ https://linqs.soe.ucsc.edu/data 
+ https://github.com/shiruipan/graph_datasets  涉及化合物，引文网络，情感网络，脑功能等五个数据库
+ **graph kernel datasets** https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
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

+ Maosong Sun(孙茂松 教授), Zhiyuan Liu(刘知远 助理研究员), Jie Zhou（周界）, 清华大学计算机科学与技术学院
+ 沈华伟 **中国科学院计算技术研究所** 
+ 唐杰
+ Kipf

## 其他资源

+ 图神经网络在视频理解中的探索 https://www.bilibili.com/video/av48201125/ 
+ 图神经网络介绍-Introduction to Graph Neural Network https://www.bilibili.com/video/av62661713
+ 唐杰-图神经网络及认知推理-图神经网络学习班  https://www.bilibili.com/video/av77934956
+ 图神经网络在线研讨会2020  https://www.bilibili.com/video/BV1zp4y117bB



# 说明

+ 本仓库仅作为学术研究使用，笔记不一定完全表述了原作者的想法，如果出现问题欢迎探讨，也建议与原作者联系。
+ 如果部分内容作者或出版社有异议，内容存在错误，请联系我进行删除或修改。


# 图传播算法

## PageRank

+ 有节点不存在外链，如果一直不断迭代计算下去，R全部都会变成0

+ 有些节点只存在指向自己的外链，则计算下去会快速的发现，最后D的R值为1，其他节点都为0。

+ 假设每个节点都有一个假想的外链指向其它任一节点，这样整个图就变成了一个强连通图了。当然，为了尽量不影响最终计算的PageRank值，节点通过假想外链传递的PageRank值会乘一个权重因子β，β一般取0.2或者更小。


## **HITS**

+ **Authority**：可理解为权威页面，一般包含高质量内容。
+ **hub**：可理解为导航页面，指向很多Authority页面。
+ 被越多的hub页面所指向的页面，内容质量越高。一个hub页面会尽可能地指向更高质量的内容页面。
+ 在迭代过程中，为了保证算法的收敛性，HITS会对Authority值与hub值分别作为均方根归一化。

## Weisfeiler-Lehman算法

+ 图的相似性
+ 同时从节点的attribute和structure两方面来做辨别。其中structure信息还是通过节点的邻居来刻画，Identifaction可以通过hashing来高效判断。
