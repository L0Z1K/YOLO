# YOLO-PyTorch
YOLO in PyTorch
## 핵심 로직 번역
### unified detection
We unify the separate components of object detection into a single neural network.
> 저희는 기존의 object detection에 사용되던 여러개의 seperate component들을 단일 neural network로 혼합합니다.

Our network uses features from the entire image to predict each bounding box.
> 저희 네트워크는 전체 이미지에서 feature를 추출해 각 bounding box들을 예측합니다.

It also predicts all bounding boxes across all classes for an image simultaneously.
> 또한 이미지에서 각 class에 대해 모든 bouding box들을 동시에 예측합니다.

This means our network reasons globally about the full image and all the objects in the image.
> 이것은 우리 네트워크가 full 이미지에 대해 폭넓게 추론한다는 것을 의미합니다.

The YOLO design enables end-to-end training and realtime speeds while maintaining high average precision.
> 

Our system divides the input image into an S × S grid.
If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
> Image를 S by S grid로 분할한다. 만약 object의 중심이 grid cell에 위치한다면, 그 grid cell은 object를 detecting하는데 책임이 있다.

Each grid cell predicts B bounding boxes and confidence
scores for those boxes.
> 각 grid cell은 B개의 bouding box들을 추측하고 각 box에 대해 **confidence score**를 측정한다.

These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts
> 이러한 confidence score는 모델이 상자에 객체가 포함되어 있다는 확신(confidence)과 상자가 예측하는 것이 얼마나 정확하다고(confidence!) 생각 하는지를 반영합니다.

Formally we define confidence as $Pr(Object) ∗ IOU_{truth}^{pred}.$
> 우리는 confidence score를 $Pr(Object) ∗ IOU_{truth}^{pred}.$로 정의합니다.

If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.
> 만약 object가 cell에 존재하지 않으면, confidence score는 0이어야 합니다. 존재한다면, 우리는 confidence score가 predicted box와 실제 값(ground truth)와의 겹치는 구역의 비율이길 바랍니다.

Each bounding box consists of 5 predictions: x, y, w, h, and confidence.
> 각 bouding box는 5가지 값의 예측으로 이루어집니다: x, y, w, h, 그리고 confidence score.

The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell.
> (x, y)는 grid cell의 경계에서 box의 무게중심까지의 상대적 좌표를 나타냅니다.

The width and height are predicted relative to the whole image.
> w, h는 이미지에 대해 상대적으로 예측됩닌다.

Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.
> 마지막으로, confidence prediction은 실제 참값과 예측된 box의 IOU를 나타냅니다.

Each grid cell also predicts $C$ conditional class probabilities, $Pr(Class_i|Object)$. These probabilities are conditioned on the grid cell containing an object.
> 각 grid cell은 $C = Pr(Class_i|Object)$라는 conditional class 확률들을 예측합니다. 이러한 확률들은 객체를 포함하는 그리드 셀에서 조절됩니다.

We only predict one set of class probabilities per grid cell, regardless of the number of boxes B.
> 하나의 grid cell에 대해 박스의 숫자인 B에 상관없이 한 set의 class의 확률을 구합니다.

At test time we multiply the conditional class probabilities and the individual box confidence predictions, $Pr(Class_i|Object) ∗ Pr(Object) ∗ IOU^{truth}_{pred} = Pr(Class_i) ∗ IOU^{truth}_{pred}$, which gives us class-specific confidence scores for each box. These scores encode both the probability of that class appearing
> test를 진행할때, 우리는 계산한 conditional class probabilities와 individual box confidence prediction을 곱합니다, $Pr(Class_i|Object) ∗ Pr(Object) ∗ IOU^{truth}_{pred} = Pr(Class_i) ∗ IOU^{truth}_{pred}$. 이 공식은 각 box에 대해 class별로 confidence score를 줍니다. 이 점수들은 class가 나타날 확률을 나타냅니다.

For evaluating YOLO on PASCAL VOC, we use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. Our final prediction is a 7 × 7 × 30 tensor.
> YOLO를 PASCAL VOC dataset에서 측정할떄, 우리는 S = 7, B = 2, C=20(pascal voc는 class가 20개). 따라서, 최종 prediction은 S*S*(B*5+C) = 7 * 7 * 30 tensor이다.
### Network design
We implement this model as a convolutional neural network and evaluate it on the PASCAL VOC detection dataset
> 저희는 이 모델을 CNN으로 model을 구현했습니다. 평가는 PASCAL VOC detection dataset에서 이루어졌습니다.

The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates.
> 네트워크의 첫 convolutional layers는 image에서 feature를 추출하는데, FC layer가 output의 확률과 좌표를 예측합니다.