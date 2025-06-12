# GFAN: An Explainable Fully Attentional Network for Multivariate Time-Series Forecasting

This is a PyTorch implementation of the paper [An Explainable Fully Attentional Network for Multivariate Time-Series Forecasting](https://doi.org/10.1016/j.knosys.2025.113780).

```
@article{WU2025113780,
title = {An explainable fully attentional network for multivariate time-series forecasting},
journal = {Knowledge-Based Systems},
volume = {324},
pages = {113780},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.113780},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125008263},
author = {Qilong Wu and Jian Xiao}
}
```

## About
Attention mechanism has achieved amazing results in various fields of artificial intelligence. However, in the field of time series forecasting, the performance of the transformer-based attention model has been questioned. At the same time, the previous transformer-based models lack interpretability to variables, which makes it difficult to be widely used. Therefore, we propose a fully attention-based model for multivariate time series forecasting, which is explainable. It consists of two components: (i) a Granger attention block that mines Granger causality between multiple variables; (ii) a skip attention block that replaces the point-wise feed-forward network. In order to verify the effectiveness of the proposed method, we conducted extensive experiments on datasets in multiple domains. The empirical results show that our Granger causality-based fully attentional network (GFAN) significantly improves the long-term prediction accuracy due to the use of the full-attention framework compared to other transformer-based models, which reflects the contribution of the attention mechanism to the predictive performance of the model.

**1. The architecture of GFAN.**

<p align="center">
<img src=".\pic\framework.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of GFAN.
</p>

**2. Granger Attention mechanism**

<p align="center">
<img src=".\pic\Granger_Attention.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Granger Attention mechansim.
</p>

## Contact
If you have any questions, please contact [qilong@stu.zuel.edu.cn](202021080121@stu.zuel.edu.cn).
