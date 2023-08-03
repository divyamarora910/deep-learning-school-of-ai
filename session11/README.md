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

## Model Definition

<img width="531" alt="Screenshot 2023-08-03 at 4 26 59 PM" src="https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/e0e5adfb-799c-420c-ba7c-5dd8af9e68f0">

## Misclassified Images
![image](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/18633485-b192-4c4c-b51f-39556155c7db)

## Gradcam
#### Layer on which gradcam is applied to
![Greenshot 2023-08-03 18 13 20](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/670bdb05-73a8-4525-ba8e-185ae2c1986e)

#### Gradcam example
![image](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/4c520b83-23fe-4506-abbc-0f60c7b47d6e)
