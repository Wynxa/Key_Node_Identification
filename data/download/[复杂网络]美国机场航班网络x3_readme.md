---
Topic:
    - 科技互联网
,    - 社交网络
,    - 旅游服务
,    - 交通出行
,    - 运输物流

Field:
    - 空间统计
,    - 数据分析
,    - 数据挖掘
,    - 数据处理

Author:
    - Shelter

Contact:
    - 2301273@stu.neu.edu.cn

Institution:
    - 东北大学

License:
    - CC-BY-SA 4.0 署名-以相同方式共享

Ext:
    - .csv

DatasetUsage:
    - 909557
---

# The network of airports in the United States
# 美国机场网络数据x3

### 1. 数据来源
[原网站](`https://toreopsahl.com/datasets/#usairports`)提供txt与dl格式下载(下文附链接)但需要一些魔法。**本项目提供处理后对应数据集的`csv`格式**，其中source、target、weight分别对应连边的起点、终点和权重。

**关键词：复杂网络，航空网络**

---

### 2. 数据说明：
🚩🚩🚩 **网络数据的读取、构建和可视化可以参考[【数据使用指南】](https://www.heywhale.com/mw/project/66f43c5c925f3bb1a229ec5d?shareby=6115d9b205bc35001742bf0a#)**

##### 2.1 USairport500 (节点数: 500, 连边数: 2980)： [txt格式](http://opsahl.co.uk/tnet/datasets/USairport500.txt) | [dl格式](http://opsahl.co.uk/tnet/datasets/USairport500.dl)    
 数据集用于Collizza等人(2007)的研究，使用美国500个最繁忙的商业机场在2002年的航班数据构建，其中的节点为商业机场，两个机场存在航班则构建连边，考虑到绝大多数航班都是往返双向的，故构建网络时简化为无向网。权重则代表两个机场之间往返的最大客流量(可用座位数量，既是最大出发量，也是最大返回量)。该网络由[the Complex Networks(已404)](http://cxnets.googlepages.com/usairtransportationnetwork)最早提出。

> Colizza, V., Pastor-Satorras, R., Vespignani, A., 2007. Reaction-diffusion processes and metapopulation models in heterogeneous networks. Nature Physics 3, 276-282.

---
##### 2.2 USairport_2010 (节点数: 1574, 连边数: 17215)：[txt格式](http://opsahl.co.uk/tnet/datasets/USairport_2010.txt) | [dl格式](http://opsahl.co.uk/tnet/datasets/USairport_2010.dl)     
数据集用于博客文章中的第一部分，基于美国交通统计局Transtats网站中2010年所有机场乘运数据构建，去除了乘运与货运数据，并对自环进行了优化

> Opsahl, T., 2011. Why Anchorage is not (that) important: Binary ties and Sample selection. Available at `http://wp.me/poFcY-Vw`.

---
##### 2.3 openflights (节点数: 2939, 连边数: 15677)：[txt格式](http://opsahl.co.uk/tnet/datasets/openflights.txt) | [dl格式](http://opsahl.co.uk/tnet/datasets/openflights.dl)     
与数据2来自同一博客，包含了两个非美国机场之间的联系，权重代表两个机场之间的航线数量。

> Opsahl, T., 2011. Why Anchorage is not (that) important: Binary ties and Sample selection. Available at `http://wp.me/poFcY-Vw`.