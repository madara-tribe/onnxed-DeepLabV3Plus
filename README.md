# Versions
```
- Python 3.7.0
- pytorch 1.7.1+cu101
- torchvision 0.8.2+cu101
- onnxruntime 1.9.0
- onnx 1.10.2
```

## Deeplabv3+ backborn(mobileNet) size
```txt
Total params: 1,811,712
Trainable params: 1,811,712
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.68
Forward/backward pass size (MB): 546.89
Params size (MB): 6.91
Estimated Total Size (MB): 555.48
```

# ONNX conversion
```
- onnx 1.9.0
- cuda driver version > 440.33
- opset 11 or more
```

# Results
```txt
Overall Acc: 0.974881
Mean Acc: 0.750294
FreqW Acc: 0.952008
Mean IoU: 0.703659
```
<b>image / target / pred</b>

<img src="https://user-images.githubusercontent.com/48679574/143726100-248e44a7-20d9-4c43-a815-1161fc2bc6b9.png" width="750px">
<img src="https://user-images.githubusercontent.com/48679574/143726114-15b2c56b-1f5c-4257-95f3-ff1c9a8d91af.png" width="750px">
<img src="https://user-images.githubusercontent.com/48679574/143726118-a142df45-034f-4fa5-a441-fe935db35b9b.png" width="750px">


