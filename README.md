# Recommender Systems
Some of the machine learning projects I worked on for my former company in helping users recommend products, movies and candidates.

I started with a very simple item-item collaborative filtering method to recommend relevant products from the superstore datasets that is provided by Tableau. You may find the data here: https://community.tableau.com/docs/DOC-1236

The collaborative filtering(CF) uses actions of users to suggest the similar items that have similar features. To implement CF, KNN is one of the simplest and effective methods to select from. KNN is known for its memory based approach which works directly with the values and searches of its nearest neighbors instantaneously, assuming no pre-built model. The Product Recommendations.py file in my repository takes records of interaction between a user and a product and calculates the similarities based on the quantity/clicks a user has performed and retrieves 5 closest neighbors to be the possible recommendations to the user.

However, before computing KNN algorithm, one might notice that with the increase in data, feature space also increases, which will require more and more resources from your machine as well as leading to overfitting in your model. In order to keep important informaiton without dropping features, PCA is one of the many unsupervised techniques that can help reduce the feature space. PCA constructs numbers of principal component, each of which contributes different variation in the data. By choosing the number of components, it will retain certain amount of information in the subspace that is being projected to higher dimension. Here, the PCA function defined in the Python file, I passed .95 as a parameter to the PCA model to select optimal number of principle components that can explain 95% of variation, which will simplify the dataset used to perform the KNN algorithm.

This is just a extremely simplified product recommendation example. There are still so much more topics needed to cover. For example, how do I deal with sparse matrix? how do I predict the rating of other products? and how do I evaluate the model?

The "movie recommendation - item-based collaborative filtering" file dives in more machine learning concepts that hasn't covered previously. In this file, I imported the movie dataset from the website. You may download it from the link here: https://grouplens.org/datasets/movielens/latest/. The dataset is quite similar to the superstore dataset I used previously instead the column names for product ID and product name has been changed movie ID and movie name respectively. The data contains 100836 ratings that were created by 610 users from 1995 to 2018. In this post, I filtered out some items since I'm only interested in recommending 2000s movies. After building the use-item rating matrix, I realized the matrix is very sparse(that is, contains mostly empty values), even we have collected data from highly engaged users and items, the sparcity is still 95% since a single viewer cannot give ratings to all the items. The sparcity is a common issue when implementing collabortive filtering which soley relies on similarity measures computed over the co-rated set of items. To combat the sparcity issue, I use a baseline prediction method that fill empty values for a user by averaging the ratings of other movies that this user has rated. This is just one of many approaches to solve the sparsity. There are other algorithms that could be a better solution, for example, the matrix factorization, which I will cover in the later post. 

The matrix is now zero sparsity and is able to compute PCA technique and KNN algorithm from the previous Python file which will eventually output the suggested items based on the similarities of this item to the others. One must notice that the traditional distance measure in KNN doesn't work well in this case because of the sparsity. For example, the two movie vectors may have many zeros in common, meaning both movies don't share similar characteristics but this doesn't make them the similar. Futhermore, the KNN does not return the predicted rating of a user but the indice and distances of the recommended movies so there is a need to generate the predicted rating which could be helpful to evaluate the accuracy of the model. To obtain the predicted ratings is simple. The predicted rating for a item k for user j can be calculated by taking the ratings the user j has already rated times the weight/similarities of the user's rated movies to the item k. Finally, we import the predicted ratings along with movie names and user ID to the recommendation table.
   
Now we have one more important task, that is to evaluate the recommender system. There are certain metrics for evaluating the model. They can be offline or online. The offline evaluations let us test the effectiveness on the dataset by looking at the prediction errors. The online evaluations attemps to evaluate the recommender system by using A/B testing, which serves a group of viewers with recommender A and the other group of viewers with recommender B and compares their Click-Through-Rate to determine which recommender to choose from. In "movie recommendation - item-based collaborative filtering" file, I focus on only offline metrics. The most common way to evaluate is to calculate the prediction error by root mean squared error (RMSE) and mean absolute error (MAE). In RMSE, the errors are squared before they are averaged so that it gives higher weight to larger errors and is useful when high variance is undesirable whereas MAE simply describes average error alone and is easier to understand.  

Using one error metric only gives us a limited view of how the recommender system performs. It will always be the best practice to evaluate some other methods. Percision and recall are classical metrics used in classification algorithms. Even though I am trying to look for the error of continuous variables, it is still possible to perform percision and recall as long as we translate our variables into a binary output. I first have to determine what movies are relevant to a user by setting a threshold of relevance. The threshold for each user varies because each user rates movies differently. When a movie is above the threshold, we can say that movie is relevant to the user, meaning it is a good recommendation for him. I also need to decide how many movies we should recommend to a user. In my case, I choose the movies with the 10 highest predicted ratings. Now, we can evaluate our model by joining recommendation table and movie dataset table that has filtered only relevant movies for each user. We can calculate the porportion of recommended movies in top 10 items that are relevant, which is known as recall. We can also calculate the the porportion of the recommended movies found in the movie dataset for a user divieded by the total number of recommendations made to that user, which is known as percision. 
