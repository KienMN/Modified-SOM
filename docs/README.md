# Documentation for SOM and LVQ
## Introduction
This is a modified version of Python code for applying SOM and LVQ algorithms in classification problems. This package also supports combining multiple networks of SOM-LVQ.
## Installation
Installing the package as a library in Python by the command.
```bash
pip install git+https://github.com/KienMN/Modified-SOM.git  
```
## Dependencies
numpy  
pandas  
scikit-learn  
## Usage
Importing suitable model class from 'detection' package. Then, creating an instance for training and predicting.  
Here is several sample lines of code. Specific documentation of each model class, attributes and methods are included in source code.
```python
from detection.competitive_network import CombineSomLvq
model = CombineSomLvq(n_rows = 10, n_cols = 10)
model.fit(X_train, y_train)
model.predict(X_test)
```
## Structure
Cloning project from github by command.
```bash
git clone https://github.com/KienMN/Modified-SOM.git
```
Project structure is shown below.
```bash
SOM/
├── MANIFEST.in   // include non-code file
├── detection     // main package
├── docs          // documentations
├── setup.py      // setup file
└── tests         // testcases files
```
## Testing
Test cases are written following format of unittest module. Run test files by command.
```bash
python3 tests/{package}/{test_file_name}.py
```