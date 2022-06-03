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
The common wisdom is that computation resources of edge devices are limited, and cannot handle  large amount of computation with reasonable latency. As a result, edge-only approaches often use lightweight models (eg. SqueezeNet) but has less accuracy than that of deeper models (VGG-19, ResNet-152). Another problem with edge-only approaches is that even if the edge device is capable of heavy computation, the memory and latency of updating a new model to the edge can require large data transfers. For instance, with AlexNet, this would require 240MB of communication from the server to the edge. In contrast, cloud-only approaches avoid these issues, yet sending raw data such as images to the cloud can cost a significant amount of money due to its large memory size.
Consequently, due to the fact that there is a tradeoff of memory size, latency, and accuracy between edge-only and cloud-only approaches, DNN partitioning is a solution that partially solves this problem. It is a feasible method that offloads some computation-intensive tasks to the cloud or edges. DNN partitioning slices the computations of the original DNN, meaning that the edge device will compute the first N layers of the DNN, whereas the cloud device will compute the rest of the layers as well as performining the final classification. The intermediate output of the Nth layer on the edge device is then sent to the cloud device through the TCP/IP protocol. Sending the intermediate output instead of the original raw data could also reduce the latency and memory size of data transmission. We will be conducting experiments by slicing the computation from different layers to obtain the optimal performance.
<img width="743" alt="Screen Shot 2022-06-03 at 4 22 39 PM" src="https://user-images.githubusercontent.com/56816585/171966138-46608f0f-4092-454f-8b8b-898ba05f87b0.png">



# 4. Evaluation and Results
slice from different layers and check th memory size being sent, and the data transmission time.
tune uncertainty threshold.
try deep ensemble and mc dropout and compare
compute overall accuracy, overall memory sent.
# 5. Discussion and Conclusions

# 6. References
