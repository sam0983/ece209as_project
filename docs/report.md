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

This section should cover the following items:

* Motivation & Objective: What are you trying to do and why? (plain English without jargon)
* State of the Art & Its Limitations: How is it done today, and what are the limits of current practice?
* Novelty & Rationale: What is new in your approach and why do you think it will be successful?
* Potential Impact: If the project is successful, what difference will it make, both technically and broadly?
* Challenges: What are the challenges and risks?
* Requirements for Success: What skills and resources are necessary to perform the project?
* Metrics of Success: What are metrics by which you would check for success?

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
The memory size per data sample can be reduced using DNN partitioning, but it still sends every sample to the cloud. Our goal is to take advantage of the characteristics of model uncertainty to reduce the samples that are sent to the cloud without losing noticable accuracy. Model uncertainty can be defined as how surprised your model is to see this data sample. If the value is low, the model is certain about it’s prediction. If the result is high, the model is not confident of its classification. In this research, two popular model uncertainty methods are implemented, namely Deep Ensemble and Monte Carlos Dropout. Experiments will be conducted to compare these two methods. Ideally, the classifications that are deemed to be certain by the model uncertainty quanitifcation method should have a higher chance of being correct. The following provides the technical details of Deep Ensemble and Monte Carlos Dropout. <br />
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
As shown from the figure below, DNN parititioning is performed between conv3_1 and conv3_2, indicating that the edge device is responsible for the computations from conv1_1 to conv3_1, whereas the cloud device computes conv3_2 to the softmax layer to output the final classfication results.
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
1. slice from different layers and check the memory size being sent, and the data transmission time.
2. try deep ensemble and mc dropout and compare
3. tune uncertainty threshold.
4. compute overall accuracy, overall memory sent.\


<br />
[Experimental Setup] 
<br />
The inference tests are all run on MacBook Air. Both the edge and cloud devices' codes are simultaneously implemented on the same MacBook. TCP/IP messaging protocal is utilized to transmit the intermediate output from the edge script to the cloud script. The data transmission is executed using Python's Pickle library as well as Socket programming. Model training is performed on Google Colab's GPU. Note that although the cloud and edge is run on Mac during inference, we automatically neglect the computation time of the cloud, as a lot of work have shown that the cloud GPU's computation time is so fast it's almost negligible.
<br />
[DNN Partitioning] 
<br />

We first investigate the data and computation characteristics of each layer in VGG19. These characteristics provide insights to identify a better computation partitioning between mobile and cloud at the layer level. The convolution (conv) and fully-connected layers (fc) are the most time-consuming layers, representing over 90% of the total execution time. Convolution layers in the middle (conv3 and conv4) takes longer to execute than the early convolution layers (conv1 and conv2). Larger number of filters are applied by the convolution layers later in the DNN to progressively extract more robust and representative features, increasing the amount of computation. On the other hand, fully-connected layers are up to one magnitude slower than the convolution layers in the network. The most timeconsuming layer is the layer fc6, a fully-connected layer deep in the DNN, taking 45% of the total execution time. <br />
 The first three convolution
layers (conv1, conv2 and conv3) generate large amounts of
output data (shown as the largest dark bars) as they apply
hundreds of filters over their input feature maps to extract
interesting features. The data size stays constant through
the activation layers (relu1 - relu5). The pooling layers
sharply reduce the data size by up to 4.7× as they summarize
regions of neighboring features by taking the maximum.
The fully-connected layers deeper in the network (fc6 -
fc8) gradually reduce the data size until the softmax layer
(softmax) and argmax layer (argmax) at the end reduce the
data to be one classification label.
<br />
<img width="691" alt="Screen Shot 2022-06-04 at 6 15 58 PM" src="https://user-images.githubusercontent.com/56816585/172030841-5a6133fd-4167-4e2f-ada9-5327f39d3f64.png">
<br />
Each bar in Figure 6a represents the end-to-end latency
of AlexNet, partitioned after each layer. Similarly, each bar
in Figure 6b represents the mobile energy consumption of
Alexnet, partitioned after each layer. Partitioning computation after a specific layer means executing the DNN on the
mobile up to that layer, transferring the output of that layer
to the cloud via wireless network, and executing the remaining layers in the cloud. The leftmost bar represents sending the original input for cloud-only processing. As partition
point moves from left to right, more layers are executed on
the mobile device thus there is an increasingly larger mobile
processing component. The rightmost bar is the latency of
executing the entire DNN locally on the mobile device.
Partition for Latency – If partitioning at the front-end, the
data transfer dominates the end-to-end latency, which is consistent with our observation in Section 4.2 that the data size
is the largest at the early stage of the DNN. Partitioning at the
back-end provides better performance since the application
can minimize the data transfer overhead, while taking advantage of the powerful server to execute the more computeheavy layers at the back-end. In the case of AlexNet using the mobile GPU and Wi-Fi, partitioning between the
last pooling layer (pool5) and the first fully-connected
layer (fc6) achieves the lowest latency, as marked in Figure 6a, improving 2.0× over cloud-only processing.
<br />
[Model Finetuning] 
<br />
The target dataset used in this research is CIFAR-10, it consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The VGG19 is pretrained on ImageNet, this dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images. The reason is that CIFAR-10 is a relatively small dataset, and many studies have shown that pretraining on a much bigger dataset yields better transfer learning results, as the generalized feature representations are similar and therefore transferrable. first train whole model for baseline. then train edge model based on mc dropout of deep ensemble. then train the cloud model with intermediate output of edge model as input. compare it with the baseline.
<br />
[Deep Ensemble] 
<br />
Experiments are conducted to compare the performances of MC Dropout and Deep Ensemble. Since the logic of the sytem is to only send data to the cloud when uncertainty is higher than a threshold, the objective should be to keep the samples sent to the cloud as low as possible while maintaining the accuracy of the edge model. We develop a strategy for choosing the superior model uncertainty quantification method. By tuning the thresholds of both approaches so that the number of data sent to the cloud is approximately the same, the one with higher accuracy in "certain" classifications is the better method. The reason for this is because if model uncertainty is low yet the classifications are not accurate, it is an indication that the calculated model uncertainty is not reliable. The figure below compares the performance of MC Dropout and Deep Ensemble.
<br />
From these results, it is shown that MC Dropout has a more reliable method, as when both methods deem approximately 6000 data samples to be certain, the accuracy of these 6000 samples is 97% using MC Dropout, whereas it is 95% using Deep Ensemble. Another notable observation from this experiment is the trade-off between accuracy and the number of data sent to cloud. It is noticed that during parameter tuning, the higher the accuracy of model uncertainty quantification, the more data are seen as uncertain and sent to cloud, increasing data transmission.
<br />
[System Evaluation] 
<br />


# 5. Discussion and Conclusions

# 6. References
