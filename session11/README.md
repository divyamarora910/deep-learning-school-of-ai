# Session 11 - CAMs, LRs, and Optimizers

This Session covers the following:

- Class activation maps
- GradCAM
- Learning Rates
- Weight updates
- Constant vs Adaptive Learning Rates
- SGD
    - Gradient Perturbation
    - Momentum & Nesterov Momentum
- RMSProp
- Adam
- Best Optimizer
- LRs
    - One Cycle Policy
    - Reduce LR on Plateau
- What kind of minima do we want?

## Structure of files
#### utils.py
This has functions:
- get_train_transforms - Transformation to be applied to training data
- get_test_transforms - Transformation to be applied to test data

#### resnet.py
Contains the model definition

#### main.py
Contains train and test utility function

#### session_11(final).ipynb
Main Notebook which clones this repo and shows the working
