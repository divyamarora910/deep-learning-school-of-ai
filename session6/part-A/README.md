# Assignment - Part A
Backpropagation
# Assignment - Part B

## Problem Statement
Rewrite the MNIST CNN assignment such that it follows the below standards:

- **99.4%** validation accuracy
- Less than **20k** Parameters
- You can use anything from above you want. 
- Less than **20 Epochs**
- Have used BN, Dropout,
- (Optional): a Fully connected layer, have used GAP.

## Solution
The solution is achieved by using the following important aspects

- MaxPooling,
- 1x1 Convolutions,
- 3x3 Convolutions,
- Receptive Field,
- SoftMax,
- Learning Rate,
- Kernels and how do we decide the number of kernels?
- Batch Normalization,
- Image Normalization,
- Position of MaxPooling,
- Concept of Transition Layers,
- Position of Transition Layer,
- DropOut
- When do we introduce DropOut, or when do we know we have some overfitting
- The distance of MaxPooling from Prediction,
- The distance of Batch Normalization from Prediction

### Architecture of the Neural Net
Following is the architecture used:

<img width="561" alt="s6_1" src="https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/2e0d39fe-b7da-4e14-b1ff-98d3acaa89e3">

### Last 5 epochs
Snapshot of the loss/accuracy of the last 5 epochs(15-20)

<img width="840" alt="Screenshot 2023-06-25 at 2 02 47 AM" src="https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/467d5281-8150-4372-94a0-f73eb22b47c9">

### Final Validation accuracy

**9946/10000 ~ 99.4%**

