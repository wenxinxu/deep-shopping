# Deep-Shopping: Content-based Clothing Image Retrieval Using Deep Features

Full blog post: [click here](https://wenxinxublog.wordpress.com/2017/01/06/deep-shopping/)

Deep learning has enormous applications in computer vision, leading to state of the art results for many classical tasks like [image classification, image segmentation](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html), etc. Moreover, the features learnt via deep learning are often far more informative than the hand-crafted features for solving other problems (Reference [1](http://papers.nips.cc/paper/3674-unsupervised-feature-learning-for-audio-classification-using-convolutional-deep-belief-networks.pdf), [2](https://papers.nips.cc/paper/5548-discriminative-unsupervised-feature-learning-with-convolutional-neural-networks.pdf)). In this blog post, we show another interesting application of these ideas to one of the most important activities in our day-to-day lives — shopping!

You may have encountered such situations: some friends or celebrities show their photos on Instagram, and you find their clothes are beautiful, and want to buy a similar one for yourself as well. However, it’s difficult to figure out their brands and where to buy them. Similar situations might also happen to any goods that you’d love to have but don’t know how to search online with just a picture. This post aims to solve this problem with deep learning!

We take clothes as our first target (partly because a good training set is available). We will design a method that takes a fashion image (either a well-posed shop image or unconstrained photos) as input, and outputs a few most similar pictures of clothes in a given dataset of fashion images, as shown below. This technique can be applied to online shopping websites, which recommends similar clothes to the users based on their query images.

![alt tag](https://github.com/wenxinxu/deep-shopping/blob/master/Theme.png)
