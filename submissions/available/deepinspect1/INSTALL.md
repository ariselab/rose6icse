Apply deepinspect to testing robust CIFAR-10 models (models from [paper1](http://papers.nips.cc/paper/8060-scaling-provable-adversarial-defenses.pdf), [paper2](https://arxiv.org/abs/1811.02625))

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

# Evaluate bias errors prediction
cd ../../reproduce
python3 confusion_bugs.py
# Evaluate bias errors prediction
python3 bias_bugs_estimate_ab_and_acd.py
python3 bias_bugs_generate_results.py
```
#### Expected outputs
1. csv files are generated without errors.  
2. precision and recall of predicting confusion errors and bias errors are outputted.
