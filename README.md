# Multi-Scale Feature Fusion Quantum Depthwise Convolutional Neural Networks for Text Classification

## Overview

## Results
<p align="center">
<img src="results/comparison_with_baselines.png" alt="Alt text"  width="40%" height="40%" />
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
  where ```model-name``` can be either 'msff-qdconv' or 'msff-dconv'. The pre-trained models are located in the ```model/RP``` directory. See ```RP.ipynb``` for a demo. 
  
- classical models
  
  Use the following code for model evaluation:
  ```
  $ python RP_c_eval.py model-name
  ```
  Use the following code for model training:
  ```
  $ python RP_c.py model-name
  ```
  where ```model-name``` can be either 'msff-dconv' or 'msff-conv'. The pre-trained models are located in the ```RP/model/RP``` directory.


The ```MC/``` directory has the same code structure.




