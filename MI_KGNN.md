### MI-KGNN

[论文](https://ieeexplore.ieee.org/abstract/document/9722999)

基于图的推荐系统，提出了四个基本假设：

​	A1：相同行为的用户是相似的

​	A2：和同一个用户连接的item是相似的

​	A3：拥有相同属性的item是相似的

​	A4：拥有相同兴趣的用户是相似的

传统的协同过滤算法，只完成了A1,A2假设，目前基于GNN的推荐算法，部分是完成了A3或者A4，本篇论文在信息传播中，让邻居节点和用户共同决定中心节点的权重，实现了A3和A4假设。

#### 引言

GNN-based method面临的两大挑战：

（1）如何获取和利用图中的交互信息

（2）如何在信息传播过程中量化不同节点的重要度

本篇论文的贡献：

1. 提出了一个新的模型：更好地表示结点和描述用户的兴趣
2. 提出了一个双注意力机制：优化信息传播
3. 在三个数据集上实验，recall提高，计算代价降低

#### 相关工作

1.图卷积神经网络

- 基于光谱
- 基于空间

2.图推荐系统

- 基于嵌入：使用KG中的结构信息来学习节点的隐式表示
- 基于路径：基于元路径提取特征，性能主要取决于手动选择的元路径
- 基于GNN：这类领域的大多数工作都是基于表示学习，侧重于学习用户偏好，然后根据节点表示来预测交互的可能性

3.图注意力网络

通常在聚合过程中引入一种注意力机制，给重要节点分配更大的权重；现有的GCN注意力机制的研究主要集中在领域节点对中心节点的重要度上。

#### 问题定义

KG = (h, r, t)

交互矩阵A

任务描述：从交互矩阵A中和KG中学习users和items的特征，然后预测user和item的交互

#### MI-KGNN

##### 模型解释

本模型主要涉及两个步骤，信息传播、信息聚合

- 信息传播

对于某个用户，通过item的邻居节点传递过来的信息判断item对于用户的重要度，即user的偏好可以根据item的邻居来表征

具体公式：
$$
v_{N}^{u} = \sum_{n_{u}^{j}\in N(v) }^{}f(u,v,r_{v,n_{v}^{i}} ,n_{v}^{j}  ).  n_{v}^{j}
$$
上述公式即是用item的邻居表征用户的偏好，公式中的v表示的就是与user直接相连的entity，nvj表示的是entity的下一跳节点，即邻居节点，rv,niv表示的是v和其邻居节点之间的关系，N(v)表示KG中v的邻居节点的集合

f函数是本篇论文提出的控制信息传播权重的双重注意机制，双重注意力机制包括用户注意力机制和邻居注意力机制

- user attention

对于一个给定的项目v，其邻居的重要性在不同用户的角度下是不同的，如何量化这种不同呢？本篇论文将这种不同量化为：用户向量和传播后的信息的内积，给定邻居节点对用户的重要性可以表示为：
$$
d_{u,n_{v}^{j} }=u^{T}.softmax(r_{v,n_{v}^{j} }   \odot n_{v}^{j})
$$
根据用户对每个邻居节点的偏好，可以将规范化的权重定义为：
$$
\alpha _{u,n_{v}^{j} }=\frac{e^{u^{T}.softmax(r_{v,n_{v}^{j} }   \odot n_{v}^{j})} }{ {\textstyle \sum_{n_{v}^{j}\in N(v) }^{}}e^{u^{T}.softmax(r_{v,n_{v}^{j} }   \odot n_{v}^{j})}  }
$$

- neighborhood attention

对于一个中心节点，它的邻居节点的重要性也是不同的，本篇论文根据邻居节点的重要性将注意力权重分配给中心节点，量化为：传输信息和中心节点向量的内积，给定邻居节点对中心节点的重要性可以表示为：
$$
d_{v,n_{v}^{j} }=v^{T}.softmax(r_{v,n_{v}^{j} }   \odot n_{v}^{j})
$$
归一化邻域注意权值可以正式定义如下:
$$
\alpha _{v,n_{v}^{j} }=\frac{e^{v^{T}.softmax(r_{v,n_{v}^{j} }   \odot n_{v}^{j})} }{ {\textstyle \sum_{n_{v}^{j}\in N(v) }^{}}e^{v^{T}.softmax(r_{v,n_{v}^{j} }   \odot n_{v}^{j})}  }
$$

- dual attention

上面两种注意机制被用来控制相同的信息向量，user attention和neighborhood attention可以以加权和的形式组合起来。双重注意机制定义如下：
$$
f(.)=\alpha _{u,v,v_{N}^{j} }=\theta . \alpha _{u,v_{N}^{j} }+(1-\theta )\alpha _{v,v_{N}^{j} }
$$
θ是一个控制两种注意机制的影响程度的超参数。同时，它也代表了用户的行为受到个人偏好的影响的程度。

- 信息聚合

聚合从邻居节点传播的信息，在信息聚合过程中增加中心节点与其相邻节点之间的信息交互，MI-KGNN中的聚合器定义为：
$$
v^{u} =agt(v,v_{N}^{u})=\delta(W_{1} (v+v_{N}^{u})+W_{2}  Q(v,v_{N}^{u})+b)
$$
w1, w2, b是参数，Q(.)表示的是两个矩阵对应位置元素进行乘积

##### 模型预测

$$
\hat{y}_{uv}  =u^{T}.v^{u}  
$$

表示user选择某个item的概率

MI-KGNN使用梯度下降算法来更新参数，loss function如下
$$
L=\sum_{u\in U }^{} (\sum_{v:y_{uv}=1 }^{} L_{c}(y_{uv},\hat{y}_{uv}   )-\sum_{i=1}^{T^{u} } E_{v_{i}\sim P(v_{i})L_{c}(y_{uv_{i} },\hat{y}_{uv_{i} }   )}) +\lambda ||W_{1}+W_{2}||_{2}^{2}
$$


