# FederationClustering

conda create -n FC python=3.14
pip install -r requirements.txt
python -u download_data.py

这是一个非常有意思的问题 — 你提出的 “联邦 + 分层 GMM 树 (hierarchical GMM tree)” 聚类方式，确实是在多个研究方向（model‑based clustering、层次 GMM、联邦 /分布式学习）交叉。虽然目前可能还没有完全与 “你所说的 T 算子 + 随机维度划分 + 递归树状 GMM” 一模一样的经典工作，但有不少相关论文可以为你提供参考和借鉴。

下面是一些参考论文 +研究方向，以及如何将这些思想映射到你描述的方法中。

---

## 一些相关或启发性论文

1. **Hierarchical Clustering of a Mixture Model** — *Jacob Goldberger & Sam Roweis* (NIPS 2004)

   * 经典的层次 GMM 聚类 (hierarchical clustering of Gaussian mixture components)：他们的方法是对一个大的 GMM 组件 (mixture) 进行归并 (merge) / 聚类，以得到一个更紧凑 (smaller) 的 GMM，同时保留原有组件结构。 ([NIPS 论文集][1])
   * 该算法只依赖模型参数 (均值、协方差等)，不必对所有数据反复采样。 ([NeurIPS Papers][2])
   * 虽然不是严格 “树形 GMM 树生成 + 自底向上 (bottom-up)” 和你说的那种迭代算子，但在 “将多个高斯组件聚成更高层结构” 上非常贴合。

2. **Hierarchical Gaussian Mixture Model with Objects Attached to Terminal and Non-terminal Dendrogram Nodes** — *Łukasz P. Olech & Mariusz Paradowski* (arXiv, 2016)

   * 他们提出一种分层 GMM，可以在树 (dendrogram) 的内部节点和叶节点都“挂 (attach)”对象 (data points)；也就是说，不只是叶子类 (clusters) 表达数据，内部节点 (higher-level clusters) 也可以有自己的成员。 ([arXiv][3])
   * 这种机制可能与你的 “如果某个高斯分布内的数据比较均匀就停止细分 (stop splitting)” 的想法比较契合 — 因为你可以通过观测每层节点 (高斯) 内部数据分布情况来决定是否继续分裂。

3. **Gaussian mixture learning via adaptive hierarchical clustering** — *J. Li 等 (2018)*

   * 他们提出了一种 **自适应分层 (adaptive hierarchical)** 方法来学习 GMM，特别是面向大规模数据。 ([科学直通车][4])
   * 虽然具体细节 (算法结构) 可能和你所设想的 “每次随机抽 M 个样本 + 维度空间随机划分 + 迭代 + 细分” 不完全同，但 “自适应层次聚类 + GMM 学习” 是非常相关的。

4. **Hierarchical Gaussian Mixture based Task Generative Model (HTGM)** — *Y. Zhang 等 (NeurIPS 2023 / ICLR)*

   * 这篇文章来自元学习 (meta‑learning) 社区：他们把任务 (task) 嵌入 (embedding) 看成来自一个 **分层 GMM (hierarchical GMM)**，即任务级别、类级别都有高斯混合分布。 ([NeurIPS 会议论文集][5])
   * 用 EM (Expectation‑Maximization) 来估计参数。 ([NeurIPS 会议论文集][5])
   * 虽然他们的 “树结构”是任务 (meta learning) 层次 (任务与类) 之间，而不是数据本身在 feature 空间上的递归细分，但他们的层次 GMM 模型思路 (hierarchical mixture, EM 学习) 对你构建类似算子 T、分支 (split) 逻辑是很好的参考。

5. **Federated Gaussian Mixture Models (FedGenGMM)** — *Sophia Zhang Pettersson 等 (arXiv 2025)*

   * 他们提出一个 one-shot 联邦 (federated) GMM 方法：各客户端分别训练本地 GMM，然后通过一种生成 (generative) 方法在服务器端合并 (aggregate)。 ([arXiv][6])
   * 这个工作正好在 “联邦 + GMM” 这个方向上，是你考虑 “联邦聚类 (federated clustering)” 很接近的文献。

6. **Personalized Federated Learning under Mixture of Distributions** — *Y. Wu 等 (2023)*

   * 他们提出了 **FedGMM (Federated Gaussian Mixture Model)** 框架，用 EM 在联邦设置下训练 GMM。 ([Proceedings of Machine Learning Research][7])
   * 他们分析了混合分布 (mixture of distributions) 情况下 federated learning 的收敛性，并提供了估计方法。

7. **Gaussian Mixture Clustered Federated Learning (GMCFL)** — *J. Pei 等 (ICLR “Clustered Federated Learning with slightly skewed labels”) *

   * 他们利用 GMM 聚类 (Gaussian mixture clustering) 对本地模型参数 (local updates) 进行分组 (聚类)，进而进行联邦学习。 ([OpenReview][8])
   * 这个是 “联邦 + 聚类 (层次可能不深，但聚类 + 客户端分组)” 的典型应用。

---

## 结合你提出的方法 (“T 算子 + 递归 + 分支细化”) 的对比与建议

* **层次 GMM + 聚类 (非联邦) 的启发**：Goldberger & Roweis (2004) 的层次聚类 GMM、Olech & Paradowski 的分层 GMM、J. Li 的 adaptive hierarchical GMM 都可以作为你 “递归 + 分裂 / 合并 GMM 组件” 的基础。
* **联邦 (federated) 的聚类 + GMM**：FedGMM (Wu 等)、FedGenGMM (Pettersson 等)、GMCFL (Pei 等) 等是联邦设置里真正做 GMM 聚类或聚类 + 模型聚合的工作。
* **任务 /元学习 (meta-learning) 的层次 GMM**：HTGM (Zhang 等) 给出了如何在 embedding 空间里做层次 GMM 模型，并且用 EM 学习，这对你设计 “每一层节点一个 GMM + 继续拆分或停止” 的策略应该很有启发。

---

## 总结（建议）

* 虽然目前没有 **精确匹配** “T 算子 + 抽 M + 随机维度划分 + 递归 GMM 树” 的经典论文（至少在公开文献中还不常见），但上述论文提供了非常有价值的思想基础。
* 你可以结合 **分层 GMM** (hierarchical GMM) 和 **联邦 GMM 聚类 (federated GMM clustering)** 两条脉络，将它们融合到你自己的算法设计。
* 在设计时，可以考虑如下策略：

  1. 在每个节点 (高斯分布) 上做局部 EM (分布估计)，判断是否需要继续拆分 (split)。
  2. 使用联邦机制 (客户端 + 服务器) 来合并 / 聚类 GMM 组件 (类似 FedGMM / FedGenGMM)。
  3. 对于是否细化 (split) 的判断，可以依据某种不纯度 (variance, entropy, 样本数) 或 BIC / AIC 的统计准则。

---

如果你愿意，我可以帮你找 **最近 (近 5 年) 在联邦 (Federated) + 层次 (Hierarchical) GMM 聚类** 方向发表的所有论文（综述级别），然后你可以更系统地对标。要我帮你查吗？

[1]: https://papers.nips.cc/paper/2585-hierarchical-clustering-of-a-mixture-model?utm_source=chatgpt.com "Hierarchical Clustering of a Mixture Model"
[2]: https://papers.neurips.cc/paper/2585-hierarchical-clustering-of-a-mixture-model.pdf?utm_source=chatgpt.com "Hierarchical Clustering of a Mixture Model"
[3]: https://arxiv.org/abs/1603.08342?utm_source=chatgpt.com "Hierarchical Gaussian Mixture Model with Objects Attached to Terminal and Non-terminal Dendrogram Nodes"
[4]: https://www.sciencedirect.com/science/article/am/pii/S0165168418301397?utm_source=chatgpt.com "Gaussian mixture learning via adaptive hierarchical ..."
[5]: https://proceedings.neurips.cc/paper_files/paper/2023/file/982ca2640e64bf7a1908b028ebc8734a-Paper-Conference.pdf?utm_source=chatgpt.com "Hierarchical Gaussian Mixture based Task Generative ..."
[6]: https://arxiv.org/html/2506.01780v1?utm_source=chatgpt.com "Federated Gaussian Mixture Models"
[7]: https://proceedings.mlr.press/v202/wu23z/wu23z.pdf?utm_source=chatgpt.com "Personalized Federated Learning under Mixture of Distributions"
[8]: https://openreview.net/pdf?id=qPwZouq5sY_&utm_source=chatgpt.com "CLUSTERED FEDERATED LEARNING WITH SLIGHTLY ..."
