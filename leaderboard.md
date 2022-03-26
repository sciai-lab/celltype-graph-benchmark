## Results
CellTypeGraph leaderboard, you are welcome to open a PR and propose a new entry to the current leaderboard.

## 5-fold Cross Validation
|             Model              |  top-1 Accuracy  | class-average Accuracy |                          source code                          |
|:------------------------------:|:----------------:|:----------------------:|:-------------------------------------------------------------:|
|  EdgeDeeperGCN **[1]** &ast;   | 0.878 &pm; 0.047 |    0.797 &pm; 0.095    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
| EdgeTransf. GCN **[2]** &ast;  | 0.868 &pm; 0.044 |   0.777 &pm; 0.0.98    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
|   Transf. GCN **[2]** &ast;    | 0.868 &pm; 0.045 |   0.779 &pm; 0.0.98    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
|      GCNII **[3]** &ast;       | 0.863 &pm; 0.050 |    0.772 &pm; 0.100    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
|    GraphSage **[4]** &ast;     | 0.859 &pm; 0.048 |    0.765 &pm; 0.093    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
|      GATv2 **[5]** &ast;       | 0.855 &pm; 0.041 |    0.757 &pm; 0.087    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
|       GAT **[6]** &ast;        | 0.824 &pm; 0.033 |    0.705 &pm; 0.084    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
|       GCN **[7]** &ast;        | 0.762 &pm; 0.043 |    0.617 &pm; 0.077    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
|       GIN **[8]** &ast;        | 0.714 &pm; 0.071 |    0.563 &pm; 0.136    | [plantcelltype](https://github.com/hci-unihd/plant-celltype)  |
 &ast; As presented in "CellTypeGraph: A New Geometric Computer Vision Benchmark, *Cerrone et al.*, CVPR, 2022".

## References:
* **[1]** Deepergcn: All you need to train deeper GCNs, *Li et al.*, arxiv, 2020. 
* **[2]** Masked label prediction: Unified message passing model for semi-supervised classification, *Shi et al.*, arxiv, 2020.
* **[3]** Simple and deep graph convolutional networks, *Zhewei et al.*, ICML 2020.
* **[4]** Inductive representation learning on large graphs, *Hamilton et al.*, NeurIPS, 2017.
* **[5]** How attentive are graph attention networks? *Brody et al.*, arxiv, 2021.
* **[6]** Graph attention networks, *Velickovic et al*, ICLR, 2018. 
* **[7]** Semi-supervised classification with graph convolutional networks. *Kipf and Welling*, ICLR, 2016. 
* **[8]** How powerful are graph neural networks?, *Xu et al.*, ICLR, 2018.