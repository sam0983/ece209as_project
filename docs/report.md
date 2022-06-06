# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Provide a brief overview of the project objhectives, approach, and results.
# 1. Introduction

  AIoT (Artiﬁcial Intelligence of Things) combines AI (Artiﬁcial Intelligence) and IoT (Internet of Things), AIoT can improve businesses and their services by creating more value out of IoT-generated data. However, there is often a trade-off between the cost and latency of transmitting data to the cloud and the accuracy of the edge device. The objective of this research project is to explore a method that may give us the best of both worlds, getting competitive accuracy while also keeping the cost and latency relatively low.<br />

  This project is mainly comprised of two research components, being DNN partitioning and model uncertainty quantification. DNN partitioning is a computation partitioning strategy that effectively leverage both the cycles in the cloud and on the mobile device to achieve low latency, low
energy consumption, and high datacenter throughput for deep learning applications such as real-time object detection. One of the notable researches regarding DNN partitioning is the work of Kang, Yiping, et al, which uses a lightweight scheduler to automatically partition DNN computation between mobile devices and datacenters at the granularity of neural network layers and finds the optimal layer to slice the computation. Although our research does not employ the automatic scheduler, Kang, Yiping et al's work thoroughly investigates the efficacy as well as the characteristics of DNN partitioning, which is very helpful for our exploration of the possibilities of DNN partitioning. Although DNN partitioning is very well-researched, a limitation of the status quo DNN partitioning approach is that it still sends all the intermediate data to the cloud, meaning that although it effectively lowers the memory per sample, the total number of samples sent are unchanged, which may still cost significantly if there is a vast amount of data. On the other hand, model uncertainty quantification is a way of conveying the uncertainty from the models's output. Some state of the art methods include Monte Carlo dropout and Deep Ensemble. <br />

  In this research, we propose a strategy that utilizes model uncertainty to intelligently calculate the confidence of the edge model classification, the system then determines the trustworthiness of the edge model to decide whether it is necessary to send the sample to the cloud for a more accurate classification using the rest of the deep neural network. To the best of my knowledge, this is the first attempt to combine DNN partitioning and model uncertainty quantification in an AIoT research project, making this a quite novel approach. According to our hypothesis, the use of model uncertainty quantification could help alleviate the limitation of DNN partitioning, and the joint traits of these topics can contribute to a more reliable and efficient AIoT system.<br />

  If the uncertainty quantification on the edge device is reliable and can serve as a gate of data transmission to the cloud, it can effectively saves cost and latency caused by excessive data transmission, while ensuring a competitive classification accuracy at the same time, which would be very beneficial for industries and academia.<br />

  The novelty of this project is also its biggest risk, since it is a very new approach, it is difficult to find related work to compare with as well as evaluate whether this method is even plausible. Furthermore, unlike status quo DNN partitioning approaches that merely slice the computations, where only the cloud performs classification, our method allows the edge model to also perform classification. As a result, it would require extensive experimentation to determine whether this research can be successful. To succeed in this research, it is required to have a thorough understanding of model uncertainty and DNN partitioning to explore the possibillities of incorporating them together effectively as well as fine-tuning the parameters to achieve the optimal performance.<br />

  There are multiple metrics to evaluate whether this system is performing well. The first metric is after adding a classifier to the first segment of the neural network, is the accuracy only using the edge device high enough? It doesn't have to be as high as that of the entire DNN, but it should at least be able to correctly classify most data. The second metric is how reliable the model uncertainty quantification is, whether outputs with uncertainties higher than a threshold are indeed inaccurate and needs to be transmitted to the cloud for further computation. The last metric is the accuracy using the second segment of the neural network (cloud device), whether the unreliable classification from the edge device can indeed be correctly classified if transmitted to the cloud. <br />

# 2. Related Work

# 3. Technical Approach
[DNN partitioning] <br />
The common wisdom is that computation resources of edge devices are limited, and cannot handle  large amount of computation with reasonable latency. As a result, edge-only approaches often use lightweight models (eg. SqueezeNet) but has less accuracy than that of deeper models (VGG-19, ResNet-152). Another problem with edge-only approaches is that even if the edge device is capable of heavy computation, the memory and latency of updating a new model to the edge can require large data transfers. For instance, with AlexNet, this would require 240MB of communication from the server to the edge. In contrast, cloud-only approaches avoid these issues, yet sending raw data such as images to the cloud can cost a significant amount of money due to its large memory size. <br />
Consequently, due to the fact that there is a tradeoff of memory size, latency, and accuracy between edge-only and cloud-only approaches, DNN partitioning is a solution that partially solves this problem. It is a feasible method that offloads some computation-intensive tasks to the cloud or edges. DNN partitioning slices the computations of the original DNN, meaning that the edge device will compute the first N layers of the DNN, whereas the cloud device will compute the rest of the layers as well as performining the final classification. The intermediate output of the Nth layer on the edge device is then sent to the cloud device through the TCP/IP protocol. Sending the intermediate output instead of the original raw data could also reduce the latency and memory size of data transmission. We will be conducting experiments by slicing the computation from different layers to obtain the optimal performance.
<br />
<img width="743" alt="Screen Shot 2022-06-03 at 4 22 39 PM" src="https://user-images.githubusercontent.com/56816585/171966138-46608f0f-4092-454f-8b8b-898ba05f87b0.png">
<br />
[Model Uncertainty Quantification] <br />
The memory size per data sample can be reduced using DNN partitioning, but it still sends every sample to the cloud. Our goal is to take advantage of the characteristics of model uncertainty to reduce the samples that are sent to the cloud without losing noticable accuracy. Model uncertainty can be defined as how surprised your model is to see this data sample. If the value is low, the model is certain about it’s prediction. If the result is high, the model is not confident of its classification. In this research, two popular model uncertainty methods are implemented, namely Deep Ensemble and Monte Carlos Dropout. Experiments will be conducted to compare these two methods. Ideally, the classifications that are deemed to be certain by the model uncertainty quantifcation method should have a higher chance of being correct. The following provides the technical details of Deep Ensemble and Monte Carlos Dropout. <br />
[Monte Carlos Dropout] <br />
Monte Carlo Dropout, or MC Dropout for short, is a technique that uses dropout layers in your model to create variation in the model’s outputs. Dropout layers are usually used as a regularization technique during training. Some neurons are randomly dropped out at a certain probability during a forward pass through the network. This has been shown to make the model more robust against overfitting. Usually, these dropout layers are disabled after training as to not interfere with the forward pass on a new image. But for MC Dropout, the dropout layers are still activated, meaning neurons can still randomly drop out. This results in a variation of the softmax results of the model. During inference, we will not only predict on the image once, but multiple times and analyze the different outputs generated by the multiple forward passes. <br />
![image](https://user-images.githubusercontent.com/56816585/171967380-1227fd2a-576c-416a-af0f-eccbda07cebb.png)

[Deep Ensemble] <br />
Instead of using one model and predicting multiple times with it, the idea is to use multiple models of the same type, randomly initialize their weights and train them on the same data. This will also create a variation in the model parameters. If the model is trained robustly and is certain about an image, it will output similar values for each forward pass. Following the initialization, all models are trained on the same training data. A number of 3 or 5 models is a good choice. To obtain the model’s uncertainty on a given data sample, it is passed through each of the models in the ensemble and its predictions are combined for analysis. <br />
![image](https://user-images.githubusercontent.com/56816585/171967499-c8fd99b6-7acd-4a59-8ccd-ebf47c26c07e.png)
<br />
After obtaining multiple outputs either using MC Dropout of Deep Ensemble, we calculate the model’s uncertainty from these outputs. Predictive entropy is used to estimate the uncertainty of the model on a given data sample.
![image](https://user-images.githubusercontent.com/56816585/171981776-8ffdf1e4-0069-4a5c-b659-86367e902e69.png)
The epsilon in the equation prevents a division by 0, which is mathematically not defined. The result of this equation yields the model uncertainty. As previously mentioned, the higher the value, the more uncertain the model is.
<br />
[Deep Learning Model] <br />
For this research, we employ the VGG19 CNN architecture because it is a well documented and effective deep learning model for image classification, so it serves as a suitable DNN model to test our hypothesis. It consists of 16 convolutional layers followed by two fully-connected layers, and finally a softmax layer for classification. The following figure shows the components of the VGG19. <br />
![Screen Shot 2022-06-03 at 9 12 34 PM](https://user-images.githubusercontent.com/56816585/171982169-7f534479-dc88-4e3d-aef1-517a207bb448.png)
<br />
As shown from the figure below, DNN parititioning is performed between conv3_1 and conv3_2 (just for illustration), indicating that the edge device is responsible for the computations from conv1_1 to conv3_1, whereas the cloud device computes conv3_2 to the softmax layer to output the final classfication results.
<br />
![Screen Shot 2022-06-03 at 9 24 28 PM](https://user-images.githubusercontent.com/56816585/171982639-7ae299b1-33fb-4c8c-95a7-fc2505a03042.png)
<br />
Adjustments are made on the edge model to enable classification. For MC Dropout, a Dropout layer and a fully-connected layer is connected to conv3_1, where the model is then fine-tuned. Subsequently, The dropout layer is activated during inference to perform MC dropout for uncertainty quantification.
If the uncertainty is within the threshold, we trust the edge model’s classification. The process is similar for Deep Ensemble, where the only difference is that the same image is run on multiple seperate models instead of running one model multiple times. The edge models of MC Dropout and Deep Ensemble are shown below. <br />
 ![Screen Shot 2022-06-03 at 9 37 07 PM](https://user-images.githubusercontent.com/56816585/171983216-09e93350-7e42-49d0-a3d1-f675bb52a435.png)
<br />
If the uncertainty exceeds the threshold, it is deemed that the edge model is not confident with its classification , so the intermediate output of conv3_1 is sent to the cloud for a more reliable classification. The cloud model is also fine-tuned with the intermediate output of the edge model as its input. The way that this is done is by initializing the weights from conv1_1 to conv3_1 with the the learnt weights of the edge model, this segment is then frozen during finetuning. The rationale behind this is that if the weights from conv1_1 to conv3_1 is frozen during finetuning, the output of conv3_1 during training will be the same as that during inference. The overall system architecture is shown below. <br />
![Screen Shot 2022-06-03 at 10 05 22 PM](https://user-images.githubusercontent.com/56816585/171984547-a128a466-180b-4797-81e3-0cf1fea28953.png)

<br />


# 4. Evaluation and Results

<br />
[Experimental Setup] 
<br />
The inference tests are all run on MacBook Air. Both the edge and cloud devices' codes are simultaneously implemented on the same MacBook. TCP/IP messaging protocal is utilized to transmit the intermediate output from the edge script to the cloud script. The data transmission is executed using Python's Pickle library as well as Socket programming. Model training is performed on Google Colab's GPU. 
<br />
[DNN Partitioning] 
<br />

We first investigate the data and computation characteristics of each layer in VGG19. These characteristics provide insights to identify a better computation partitioning between mobile and cloud at the layer level. The convolution (conv) and fully-connected layers (linear) are the most time-consuming layers, representing over 90% of the total execution time. Larger number of filters are applied by the convolution layers later in the DNN to progressively extract more robust and representative features, increasing the amount of computation. On the other hand, fully-connected layers are up to one magnitude slower than the convolution layers in the network. The most timeconsuming layer is the layer linear1, a fully-connected layer deep in the DNN, taking 40% of the total execution time. <br />
 The first four convolution
layers (conv1_1, conv1_2, conv2_1 and conv2_2) generate large amounts of
output data (shown as the largest blue bars) as they apply
hundreds of filters over their input feature maps to extract
interesting features. The data size stays constant through
the activation layers. The pooling layers
sharply reduce the data size by up to 4.3× as they summarize
regions of neighboring features by taking the maximum.
The fully-connected layers deeper in the network (linear1 -
linear2) gradually reduce the data size until the softmax layer
(softmax)at the end reduce the
data to be one classification label. Time in the following graph refers to the time it takes for transferring the output of that layer
to the cloud via wireless network. It is also observed that the data transmission time follows a similar trend of the corresponding output size, which is reasonable because data size should be positively related to its transmission time.
<br />

![dnn_plt](https://user-images.githubusercontent.com/56816585/172106836-0859e228-e691-4272-b2f8-ed6fe8f1a4c1.png)
![latency](https://user-images.githubusercontent.com/56816585/172106850-3d827993-254b-40cd-ae39-0c286dc74cea.png)

<br />
The leftmost bar (input) represents sending the original input for cloud-only processing. As partition
point moves from left to right, more layers are executed on
the mobile device. The rightmost bar represents
executing the entire DNN locally on the mobile device.
If partitioning at the front-end, the
data transmission time is very high, which is consistent with our observation that the data size
is the largest at the early stage of the DNN. Partitioning at the
back-end provides better performance since the application
can minimize the data transfer overhead. In the case of VGG19, taking latency and data transmission size into considerationn, partitioning between the pooling layer after conv4_4 and conv5_1 is the optimal choice, reducing data transmission size by 34% over cloud-only processing. In addition, we discover that the edge model size is only 40.7 MB, while the size of the entire VGG19 is 548.5 MB, making over-the-air updates much more plausible compared to edge-only processing. 
<br />
[Model Finetuning] 
<br />
The target dataset used in this research is CIFAR-10, it consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The VGG19 is pretrained on ImageNet, this dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images. The reason is that CIFAR-10 is a relatively small dataset, and many studies have shown that pretraining on a much bigger dataset yields better transfer learning results, as the generalized feature representations are similar and therefore transferrable. For all finetuning processes, SGD is used for optimization, with learning rate = 0.001, momentem = 0.9, and 5 epochs. The hyperparameters are kept constant during all finetuning processes. First, The unpartitioned VGG19 model is trained and is used as the baseline. The partitioned edge model and cloud model are also finetuned seperately, note that they are finetuned depending on MC Dropout or Deep Ensemble. For Deep Ensemble, three seperate models with randomly intialized weights are finetuned. The following table shows the finetuning results.

![Screen Shot 2022-06-06 at 12 14 58 AM](https://user-images.githubusercontent.com/56816585/172114950-aae52148-d2da-4450-9085-bd4cd2699ec2.png)

<br />

It can be seen from the table that the edge models all have accuracies of 79- 80%, which is a noticable drop compared to the baseline. This is an expected outcome because it has fewer layers than the unpartitioned VGG19. An interesting finding is that all of the cloud models, regardless of the edge models, all achieve an accuracy of 85% (1% lower than baseline). This shows that making minor modifications to the model and freezing the weights from conv1_1 to conv4_4 does not cause a significantly negative impact on the classification performance.

<br />
[Model Uncertainty Quantification] 
<br />
Experiments are conducted to compare the performances of MC Dropout and Deep Ensemble. Since the logic of the sytem is to only send data to the cloud when uncertainty is higher than a threshold, the objective should be to keep the samples sent to the cloud as low as possible while maintaining the accuracy of the edge model. We develop a strategy for choosing the superior model uncertainty quantification method. By tuning the thresholds of both approaches so that when the accuracy in "certain" classifications is approximately the same, the one with less number of data sent to the cloud  is the better method. The reason for this is because if model uncertainty is low yet the classifications are not accurate, it is an indication that the calculated model uncertainty is not reliable. The figure below compares the performance of MC Dropout and Deep Ensemble. The model uncertainty threshold (predictive entropy) is denoted as p.

![Screen Shot 2022-06-06 at 1 45 13 AM](https://user-images.githubusercontent.com/56816585/172127725-45417cfb-fb39-425a-b3ba-3b3013aec96c.png)

From these results, it is shown that although both approaches have good performances, MC Dropout is a relatively more reliable method, as when both methods achieve 96.6% accuracy on classifications that are deemed to be certain, there are  4581 uncertain classifications using MC Dropout, whereas there are 4841 using Deep Ensemble. This means when using MC Dropout, the data samples sent to the cloud are 260 less than that of using Deep Ensemble. Another reason to choose MC Dropout is that it only requires one model, while Deep Ensemble needs multiple models, which occupies more memory space. Furthermore, a notable observation from this experiment is the trade-off between accuracy and the number of data sent to cloud. It is noticed that during parameter tuning, the higher the accuracy of model uncertainty quantification, the more data are seen as uncertain and sent to cloud, increasing data transmission.
<br />
[System Evaluation] 
<br />
tune the threshold and evaluate the overaall memory sent as well as the overall accuracy, compare it to baseline.
After extensive experimentation, we have established the overall structure of the system. The edge model consists of conv1_1 to the maxpool layer after conv4_4 of the VGG19, while the cloud model consists of conv5_1 to the final sofmax layer. MC Dropout is used for model uncertainty quantification to distinguish certain and uncertain classifications. For the final evaluation, one of the metrics of success will be the overall accuracy, being the combined accuracy of the edge model's classifications on certain data samples and the cloud model's classifications on uncertain data samples. The second metric is the total number of data samples sent to the cloud, because reducing data sent to the cloud also saves data transmission cost. The ultimate goal is to achieve an accuracy comparable to that of the unpartitioned VGG19 while reducing the amount of data sent to the cloud. The results are shown in the table below.
![Screen Shot 2022-06-06 at 2 21 43 AM](https://user-images.githubusercontent.com/56816585/172133613-61017aa0-b8f2-4376-84d4-f074d015c8ff.png)
From the table above, it is seen that the accuracy of our system matches that of the baseline when p=0.01, and it deteriorates as p increases. The cause of this is quite intuitive, as when the tolerance for uncertainty is low, the ones deemed to be certain have a higher chance of being correct. However, there is also a trade-off between accuracy and data transfer as seen in the table, where good accuracy comes with the expense of sending more data to the cloud for further computation.

# 5. Discussion and Conclusions

# 6. References
