# Shopee Challenge (Beginner Category)
This repository contains the models I have used for the NDSC Shopee Challenge. I actually registered for the advanced challenge, but due to time constraints, I decided to work on the beginner category which was less time consuming to study and implement. I submitted under a friend's team "Small Data" in the beginner category, as I am not actually in that category of the competition.

Due to time constraints, I skipped the EDA by looking at the public kernel, https://www.kaggle.com/chewzy/eda-for-ndsc-2019. The data for the beginner category was straightforward to use, as it only contained the title and the images. According to the EDA, the title alone was a decent enough feature to predict the labels, but more could be done with image feature extraction for the fashion categories.

As image feature extraction required a significant amount of time, I skipped it and attempted to create models solely on the titles.

# Models
Feature preprocessing was minimal, as there was only one feature (the title), and I simply got rid of the non-letters. I considered removing stop words, but I soon realised that majority of the text contained Bahasa, which required translation. Due to API and time limits, I decided to leave that alone as well. I created word embeddings using Word2Vec as well.

I attempted to handle the classification using Multinomial Naive Bayes and Linear SVM, but the accuracies were low even after regularization. Thereafter, I proceeded to build neural networks to come up with a model that gave higher prediction scores. 

Initially, I used a simple sequential model with some Dense layers with normalisation and dropouts, but I soon realised that these models plateaued off at around 0.7 accuracy very quickly.

Then, I used LSTM to handle the classification, which gave better results probably due to the handling of the vanishing gradient I assume - I have yet to truly understand why it works better. Still, there is room for improvement. The LSTM overfitted to the training set, and from there I realised more regularization should be in place. Also, due to a bad configuration of the model checkpoint saving, I did not manage to compare training accuracy with validation accuracy at every iteration (epoch). More ideally, I would select the model weights at the point where the training and validation accuracies start to cross over.
