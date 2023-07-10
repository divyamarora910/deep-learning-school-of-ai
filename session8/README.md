# SESSION 8 - BATCH NORMALIZATION & REGULARIZATION

The assignment aims to make the network
C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10

3 versions of the above code (in each case achieve above 70% accuracy):
- Network with Group Normalization
- Network with Layer Normalization
- Network with Batch Normalization

## model.py
This contains the models used for the above three variations

### Network with Group Normalization
Represented by class ```GroupNormNet```
##### Architecture Representation
![Screenshot 2023-07-11 at 2 04 26 AM](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/81b7cf5c-29ab-4c75-a27c-f0baf3af211a)


### Network with Layer Normalization
Represented by class ```LayerNormNet```
##### Architecture Representation
![Screenshot 2023-07-11 at 2 03 57 AM](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/138be260-412a-42e7-8224-39066a83afde)

### Network with Batch Normalization
Represented by class ```BatchNormNet```

##### Architecture Representation
![Screenshot 2023-07-11 at 2 03 35 AM](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/4275fbdb-df45-4168-bedb-52997ecb7844)
              
## Accuracies
Logs of last 5 epochs
### Network with Group Normalization

### Network with Layer Normalization
![Screenshot 2023-07-11 at 2 14 42 AM](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/b340bbeb-7fbc-42ed-95ac-a3beef97f4cb)

### Network with Batch Normalization
![Screenshot 2023-07-11 at 2 14 14 AM](https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/8463c64d-0526-44fc-af68-9d4289b5c76f)

