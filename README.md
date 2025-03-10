# Multi-Scale Feature Fusion Quantum Depthwise Convolutional Neural Networks

## Overview

This is the official implementation of the MSFF-QDConv model proposed in the paper "[Multi-scale feature fusion quantum depthwise Convolutional Neural
Networks for text classification](https://doi.org/10.1016/j.enganabound.2025.106158)". The MSFF-QDConv model is a QNN model based on the quantum depthwise convolution, quantum embedding, and multi-scale feature fusion mechanism. The MSFF-QDConv model establishes a new state-of-the-art test accuracy of 96.77% on the RP dataset. Due to the use of the quantum depthwise convolution, the MSFF-QDConv model has fewer parameters and lower computational complexity compared to conventional QCNN models. Please refer to the original paper for the details of the model.

<p align="center">
  <img src="images/architecture.png" alt="Alt text" width="80%" height="80%" />
</p>
<p align="center">
  The architecture of the MSFF-QDConv model
</p>

## Results
<p align="center">
<img src="images/comparison_with_baselines.png" alt="Alt text" width="50%" height="50%" />
</p>

## Requirements
- numpy==1.23.5
- pandas==2.0.3
- pyqpanda==3.8.3.2
- pyvqnet==2.11.0
  
Use the following code to install pyvqnet:
```
$ pip install pyvqnet --index-url https://pypi.originqc.com.cn
```
When installing pyvqnet, pyqpanda will be automatically installed as a dependency.

## Usage
Take the ```RP/``` directory for example. 

- quantum models
  
  Use the following code for model evaluation:
  ```
  $ python RP_q_eval.py model-name
  ```
  Use the following code for model training:
  ```
  $ python RP_q.py model-name
  ```
  The command-line argument ```model-name``` can be either 'msff-qdconv' or 'msff-dconv'. The pre-trained models are located in the ```model/RP``` directory. See ```RP.ipynb``` for a demo of model training. 
  
- classical models
  
  Use the following code for model evaluation:
  ```
  $ python RP_c_eval.py model-name
  ```
  Use the following code for model training:
  ```
  $ python RP_c.py model-name
  ```
  The command-line argument ```model-name``` can be either 'msff-dconv' or 'msff-conv'. The pre-trained models are located in the ```RP/model/RP``` directory.


The ```MC/``` directory has the same code structure.


## Citation

```
@article{chen2024multi,
  title={Multi-scale feature fusion quantum depthwise Convolutional Neural Networks for text classification},
  author={Chen, Yixiong and Fang, Weichuan},
  journal={Engineering Analysis with Boundary Elements},
  volume={174},
  pages={106158},
  year={2025},
  publisher={Elsevier}
}
```



