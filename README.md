# Amazon-fine-food-reviews
I have downloaded the data set from Kaggle(.csv) file. The (.csv) file contains collection of text documents(568,454 food reviews) up to and including October 2012.

Our task is to predict the sentiment(positive or negative) of a reviewer’s score on a scale of 1 to 5, where 1 indicates the reviewer extremely dislikes the food he or she mentions in the review and 5 indicates the user likes the food a lot.
 I extracted the ‘Score’ and ‘Text’ column . While I was curious to know the counts for each rating, I found that the rating 5 was quite high in number as compared to other number.
 Immediately, I realized that my data is imbalanced. Now, as a result when I train my algorithm with this data, there is a possibility of miss classification as my data is biased towards higher rating.
 
 Data Preprocessing

So, my next task was to balance the data. For balancing an imbalanced data set we can mainly follow 3 approaches:

    Random under sampling: Random Under sampling aims to balance class distribution by randomly eliminating majority class examples. This is done until the majority and minority class instances are balanced out. In this approach, we reduce the data from higher class (data with 4 and 5 rating) to match the data with lower class(data with 1 and 2 rating).
    Random over sampling: Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.
    Synthetic Data Generation: In simple words, instead of replicating and adding the observations from the minority class, it overcome imbalances by generates artificial data. It is also a type of oversampling technique. In regards to synthetic data generation, synthetic minority oversampling technique (SMOTE) is a powerful and widely used method.

In this project, I used Random under sampling technique to balance the data set.


Since, rating 3 does not make much of a difference, as in we cannot predict whether the user gave a positive or a negative sentiment, I decided to eliminate rating 3 from consideration.

Splitting the data

While my data set is now balanced my next task was to split the data into training and test set. My input is ‘Text’ and my output would be ‘Sentiment’.

Count Vectorizer

After splitting the data, I have used CountVectorizer() to convert our text documents to matrix of token counts. This configuration tokenize the strings and convert them to lower case and build a vocabulary of comma separated tokens.

The data is further transformed to convert it into a matrix where our rows are the text documents and columns are the words. This data could further be interpreted to an array where the frequency of each word in a single text could be counted.

In short, CountVectorizer helps us to find the frequency of a particular word in a text document.

TfidfVectorizer

This data is further processed by applying Tfidf Vectorizer, which helps us to give more weight-age to important words which less important words for the case study would be given more weights.

Since, our code is based on counting the frequency of each word in the document, so if certain words like ‘the’, ‘if’ etc. which are present more frequently then words which are more important such as ‘buy’,’product’ etc. , which gives us the context.

n-grams

I decided to make the model bit more interesting by fitting our model with ‘n-grams’. This would further improve our model for example it would help us to differentiate between ‘good’ and ‘not good’ as it would take both words together(for bi gram count pairs). Also, it would help us to work with more features. I have set the n-grams in the range of 1–2 which helps us to extract features for 1 and 2 grams.

Fitting the model

After the pre-processing part, the model has to be fitted. For fitting this model, I have decided to work with Logistic Regression and Multinomial Naive Bayes Algorithm. I would like to compare both the models.

-> Multinomial Naive Bayes Algorithm

Text Processing works good with Multinomial NB Algorithm. So, I decided to apply Multinomial NB .
After fitting the model with my training data, I have also predicted the accuracy of the model with ‘AUC’(Area under the curve). It gives us a score of 0.924 which is good.

-> Logistic Regression

Since Logistic Regression works best with high dimensional sparse data, I have decided to fit my training data with Logistic Regression.

The ‘AUC’ score (0.9428) seems to give better prediction for Logistic Regression than Multinomial NB.

Testing the model

Finally, I decided to test my model with random sentences.

As per my project, I found that Logistic Regression has given me better result compared to Multinomial NB algorithm. But, I would also like to mention that I have used under sampling technique which means that I have not worked with entire data set. The accuracy might further improve when the entire data set is taken into consideration. It would be interesting to work with the entire data set using Deep Learning techniques such as RNN which is beyond the scope of my current work.
