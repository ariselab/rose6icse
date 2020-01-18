# DeepInspect: Testing DNN Image Classifier for Confusion & Bias Errors  (ICSE'20)

DeepInspect is a tool to test any DNN based image classifier and outputs potential confusion and bias errors.
Check out [DeepInspect website](https://github.com/ARiSE-Lab/DeepInspect) for more information details. A pre-print of the paper can be found at [ICSE20_DeepInspect.pdf](https://yuchi1989.github.io/papers/ICSE20_DeepInspect.pdf). 

## Reproduce paper results

### Requirement
OS: Ubuntu 18.10   
Python 3.7 or Python 3.8   
Need python package(can be installed by pip or conda): numpy, scipy, matplotlib, sklearn, pandas   

### Reproducing results in Table 3 and Figure 6 in paper:  
```
cd reproduce
python3 confusion_bugs.py
```

### Reproducing results in Table 4 and Figure 10 in paper:
```
cd reproduce
python3 bias_bugs_estimate_ab_and_acd.py
python3 bias_bugs_generate_results.py
```


## Apply deepinspect to testing robust CIFAR-10 models (models from [paper1](http://papers.nips.cc/paper/8060-scaling-provable-adversarial-defenses.pdf), [paper2](https://arxiv.org/abs/1811.02625))

#### Prerequisite
OS: Ubuntu 18.10  
Python 2.7 with library numpy-1.16, tqdm-4.41, torch-1.3.1, torchsummary  
Python 3.7/3.8 with library numpy, scipy, matplotlib, sklearn, pandas

#### Clone the Repository
git clone https://github.com/ARiSE-Lab/DeepInspect.git

#### Run deepinspect on robust training CIFAR-10 models to generate csv files
```
cd deepinspect/robust_cifar10/
python2 cifar10_small_deepinspect.py
```
#### Evaluate the predictions of confusion errors and bias errors.
```
cp robust_cifar_small_neuron_distance_from_predicted_labels_test_90_0.5.csv ../../data/robust_cifar10_small/robust_cifar_small_neuron_distance_from_predicted_labels_test_90.csv
cp robust_cifar_small_objects_directional_type1_confusion_test_90.csv ../../data/robust_cifar10_small/objects_directional_type1_confusion_test_90.csv
cp robust_cifar_small_test_labels_90.csv ../../data/robust_cifar10_small/test_labels_90.csv
cp robust_cifar_small_test_predicted_labels_90.csv ../../data/robust_cifar10_small/test_predicted_labels_90.csv

# Evaluate confusion errors prediction
cd ../../reproduce
python3 confusion_bugs.py
# Evaluate bias errors prediction
python3 bias_bugs_estimate_ab_and_acd.py
python3 bias_bugs_generate_results.py
```
## Expected outputs
1. Reproducing scripts should generate exactly same results as in paper.
2. When inspecting CIFAR-10 models, csv files are generated without errors.  
3. After overriding original csv files, the precision and recall of predicting confusion errors and bias errors are outputted.
