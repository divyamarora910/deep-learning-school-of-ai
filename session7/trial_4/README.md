# Trial 4

## Summary

### Target:

- Changed the out channel from 16 to 14 in multiple sequential Conv blocks.
- Added another Convolution layer before GAP
- Changed LR to 0.2 

### Results:
Parameters: 7978
Best Train Accuracy: 99.25
Best Test Accuracy: 99.45 (13th Epoch)
### Analysis:
Finding a good LR schedule is hard. We have tried to make it effective by reducing LR by 10th after the 6th epoch.
It did help in getting to 99.4 or faster, but the final accuracy is not more than 99.5. Possibly a good scheduler can do wonders here!

## Model Summary

### Model Parameter

<img width="536" alt="Screenshot 2023-06-17 at 4 54 25 AM" src="https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/1fc0f4fd-cc8f-4ab4-a6e4-ad216ed43a62">

### Model Training/testing Stats

<img width="1254" alt="Screenshot 2023-06-17 at 4 55 36 AM" src="https://github.com/divyamarora910/deep-learning-school-of-ai/assets/22102468/f18566eb-ee60-4e77-8e4c-59774df16fb7">
