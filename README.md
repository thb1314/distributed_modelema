# Distributed ModelEMA

启发于ZeroOptimizer2的原理，本项目给出一种分布式ModelEMA的实现，相比于原有的ModelEMA实现，其可以将待计算的参数均匀分配到每个计算卡上，再进行计算，极大节省了计算量。由于EMA更新过程不需要进行模型评估，所以仅需在进行模型评估之前同步所有节点的EMA Model参数。

Inspired by ZeroOptimizer2, this project presents a distributed implementation of ModelEMA. 
Compared to the its original implementation, it evenly distributes the parameters to be calculated across each computing card for processing, which greatly reduces computational cost. Since the EMA update process does not require model evaluation, we only need to synchronize the EMA Model parameters across all nodes before model evaluation step. Without bells and whistles, the distributed ModelEMA proposed in this project can seamlessly replace its original version, greatly boosting the efficiency of ModelEMA step in training pipeline.
