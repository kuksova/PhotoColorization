# PhotoColorization
Deep Learning Photo Colorization and color space learning. Exploring research literature and implementing prototypes using Pytorch

## Simple approach
Basen on LAB color space.
![image](https://user-images.githubusercontent.com/14224692/182465323-157300ae-3533-442b-a09a-58f2821203ed.png)

##  Exposure idea (inspiring Lightroom histogram)

Amount of light on three zones (dark, mid, light) in RGB channels simultaneously. Working with the prediction of color depends on light/color zone.

![image](https://user-images.githubusercontent.com/14224692/186225319-111fe40d-b5a0-4761-8de1-4b7b714dff22.png)

Results: simple convNN (7 layers)
![image](https://user-images.githubusercontent.com/14224692/186225248-666a53b9-1066-4cc2-9730-f3439e7bd6f6.png)

