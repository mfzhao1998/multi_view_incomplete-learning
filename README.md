# multi_view_incomplete_learning
Pytorch implementation for "Towards robust classification of multi-view remote sensing images with partial data availability".

<img src="https://github.com/mfzhao1998/multi_view_incomplete_learning/blob/main/Framework.png" width="75%">

This framework not only fully mines the features of multi-view images and  improves performance under complete views, but also maintains robustness under missing view without relying on restore and retrieval data.

## Illustration of code
1.Get the initial weughts for student model and teacher model through train.py;

2.The teacher model and student model learn from each other through train_ml.py.
