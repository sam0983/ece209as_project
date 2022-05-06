# Project Proposal

## 1. Motivation & Objective

AIoT (Artiﬁcial Intelligence of Things) combines AI (Artiﬁcial Intelligence) and IoT (Internet of Things), AIoT can improve businesses and their services by creating more value out of IoT-generated data. However, there is often a trade-off between the cost and latency of transmitting data to the cloud and the accuracy of the edge device. The objective of my research project is to explore a method that may give us the best of both worlds, getting competitive accuracy while also keeping the cost and latency relatively low.

## 2. State of the Art & Its Limitations

My project is mainly comprised of two research components, which is DNN partitioning and model uncertainty quantification. One of the notable researches regarding DNN partitioning is "Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge", which uses a lightweight scheduler to automatically partition DNN computation between mobile devices and datacenters at the granularity of neural network layers and finds the optimal layer to slice the computation. If I were to say that there is a limitation to DNN partitioning, it would be that it still sends all the intermediate data to the cloud, which may still cost a lot of money if there is a vast amount of data. For model uncertainty quantification,one of the state of the art methods is the Monte Carlo dropout, it uses dropout as a regularization term to compute the prediction uncertainty. I think that utilizing it smartly could help alleviate the limitation of DNN partitioning mentioned above.

## 3. Novelty & Rationale

To the best of my knowledge, this is the first time that anyone has attempted to combine DNN partitioning and model uncertainty quantification in an AIoT research project, making this a rather novel approach. According to my hypothesis, the joint traits of these topics can contribute to a more reliable and efficient AIoT system.

## 4. Potential Impact

If the uncertainty quantification on the edge device is reliable, it will be able to effectively determine whether or not it is neccessary to transmit the data to the cloud for a more accurate classification using the rest of the deep neural network. This not only saves cost and latency caused by excessive data transmission, but it could also ensure a competitive classification accuracy at the same time, which would be very beneficial for industries and academia.

## 5. Challenges

The novelty of this project is also its biggest risk, since it is a very new approach, it is difficult to find related work to compare with as well as evaluate whether this method is even plausible, so it would require extensive experimentation to determine whether this research can be successful.

## 6. Requirements for Success

To succeed in this research, it is required that I have a thorough understanding of model uncertainty and DNN partitioning to explore the possibillities of incorporating them together effectively as well as fine-tuning the parameters to achieve the optimal performance.

## 7. Metrics of Success

There are multiple metrics to evaluate whether this system is performing well. The first metric is after adding a classifier to the first segment of the neural network, is the accuracy only using the edge device high enough? It doesn't have to be perfect but it should at least be able to correctly classify most data. The second metric is how reliable the model uncertainty quantification is, whether outputs with uncertainties higher than a threshold are indeed inaccurate and needs to be transmitted to the cloud for further computation. The last metric is the accuracy using the second segment of the neural network (cloud device), whether the unreliable classification from the edge device can indeed be correctly classified if transmitted to the cloud.

## 8. Execution Plan

There are three key tasks, and they are described in the following:
1. Performind DNN partitioning and ensuring that the model still works accordingly after dividing it to the edge and the cloud.
2. Adding the model uncertainty component to the edge and testing whether it is reliable.
3. Training the edge segment and the cloud segment so that both ends could achieve optimal performance.

It will all be implemented by Liang-Chien Liu.
## 9. Related Work
### 9.a. Papers

1. VGG-19 based on the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" is used as the deep learning model in this project, as it is a very deep model and it is also one of the state of the art image classification models, making it suitable for this project.
2. For DNN partitioning, I mostly referenced the paper "Neurosurgeon: Collaborative intelligence between the cloud and mobile edge", where I learned a lot about the implementation and characteristics of DNN partitioning.
3. I referenced "Dropout as a bayesian approximation: Representing model uncertainty in deep learning" for model uncertainty quantification, as MC dropout is a relatively straightforward and effective method.

### 9.b. Datasets

List datasets that you have identified and plan to use. Provide references (with full citation in the References section below).
CIFAR-10 dataset is used in this research.

### 9.c. Software

List softwate that you have identified and plan to use. Provide references (with full citation in the References section below). <br />
Pytorch, Python, numpy

## 10. References

1. Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
https://arxiv.org/abs/1409.1556
2. Kang, Yiping, et al. "Neurosurgeon: Collaborative intelligence between the cloud and mobile edge." ACM SIGARCH Computer Architecture News 45.1 (2017): 615-629. https://dl.acm.org/doi/abs/10.1145/3093337.3037698
3. Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016. http://proceedings.mlr.press/v48/gal16.html?ref=https://githubhelp.com
4. CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html
5. Pytorch https://pytorch.org/
6. Python https://www.python.org/
7. numpy https://numpy.org/
