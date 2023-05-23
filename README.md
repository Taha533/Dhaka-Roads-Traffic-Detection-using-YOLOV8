# Dhaka Roads Traffic Detection using YOLOv8

This project aims to address the unique challenges of automated traffic detection in the context of Dhaka, Bangladesh. The project utilizes the YOLOv8 model to detect and recognize various types of vehicles on the roads of Dhaka.

## Overview
The traffic scenario in Dhaka presents complex challenges for automated traffic detection systems. This project focuses on developing an AI-based solution to detect and recognize traffic vehicles. The dataset used in this project consists of vehicle images, each containing one or more vehicles from a set of 21 different classes. This diverse dataset enables the detection and recognition of multiple types of vehicles.

The considered vehicle classes are as follows:
- Ambulance
- Auto-rickshaw
- Bicycle
- Bus
- Car
- Garbage van
- Human hauler
- Minibus
- Minivan
- Motorbike
- Pickup
- Army vehicle
- Police car
- Rickshaw
- Scooter
- SUV
- Taxi
- Three-wheelers (CNG)
- Truck
- Van
- Wheelbarrow

Dataset Link: https://www.kaggle.com/datasets/aifahim/dhakaaiyoloformatteddataset

## Model and Training
The YOLOv8 model, a state-of-the-art object detection algorithm, is employed in this project for traffic detection. The model is trained using a custom dataset created specifically for Dhaka traffic. The training process involves optimizing the model's weights through multiple iterations called epochs.

The model in this project has been trained for 10 epochs, during which it learns to accurately detect and classify the different vehicle classes. The weights obtained after training are used for performing traffic detection on various images.




## Getting Started
To train YOLOv8 on your custom dataset, follow these steps:

1. Install ultralytics:
```bash
  !pip install ultralytics
```
2. Train the model by following these lines of code:

```bash
  from ultralytics import YOLO
  model = YOLO()
  model.train(data = [path of the yaml file], epochs = no. of epochs)
```  
3. To perform detection on the test data, follow this:

```bash
  !yolo 'detect' 'predict' model= [path of the best performing weights of the model namely best.pt] source = [path of the test images]
```  
The best.pt file typically represents the weights of the model at the point where it achieved the best performance on a validation set or during training.

## Output Results:

1. First image:\
![Image 1](https://github.com/Taha533/Dhaka-Roads-Traffic-Detection-using-YOLOV8/blob/main/Output/test1.png?raw=true)

2. Second Image:\
![Image 1](https://github.com/Taha533/Dhaka-Roads-Traffic-Detection-using-YOLOV8/blob/main/Output/test2.png?raw=true)

3. Third Image:\
![Image 1](https://github.com/Taha533/Dhaka-Roads-Traffic-Detection-using-YOLOV8/blob/main/Output/test3.png?raw=true)

## Contributing
Contributions to this project are welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request.

## License
This project is licensed under the [MIT License](https://github.com/Taha533/Dhaka-Roads-Traffic-Detection-using-YOLOV8/blob/main/LICENSE). You are free to modify, distribute, and use the code for personal and commercial purposes.

## Acknowledgements
Thanks to the following resources:

- YOLOv8: https://github.com/ultralytics/ultralytics
- Dhaka Traffic Dataset: https://www.kaggle.com/datasets/aifahim/dhakaaiyoloformatteddataset

I express my gratitude to the authors and contributors of these resources for their valuable work.

## Contact
For any questions, concerns, or feedback regarding this project, please feel free to reach out to me:

Name: Bahauddin Taha\
Email: taha15@cse.pstu.ac.bd\
Thank you for your interest in this project!
