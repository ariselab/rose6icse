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
