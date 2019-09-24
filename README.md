# Recommender Systems
Some of the Python sample code I worked on for my former company in helping users  recommend products, movies and candidates.

I started with very simple item-item collaborative filtering method to recommend relevant products from the superstore datasets that is provided by Tableau. You may find the data here: https://community.tableau.com/docs/DOC-1236

The collaborative filtering(CF) uses actions of users to suggest the similar items that may have similar features. To implement CF, KNN is one of the simplest and effective methods to select from. KNN is known for memory based approach,  works directly with the values and searches of its nearest neighbors instantaneoulsy, assuming no pre-built model. The Product Recommendations.py file in my repository takes records of interaction between a user and a product and calculates the distance based on the quantity/clicks a user has performed and retrieves 5 closest neighbors to be the possible recommendations to the user. 

However, one might notice that with the increase in data, feature space also increases, which will require more and more memory from your machine as well as lead to overfitting in your model. In order to reduce the collected features, PCA is one of the many unsupervised techniques that can help reduce the feature space. PCA constructs numbers of principal component, each of which contributes different variation in the data. By choosing the number of components, it will retain different amount of information in the subspace that is being projected to higher dimension. Here, the PCA function defined in the file, I passed .95 as a parameter to the PCA model to select optimal number of principle components that can explain 95% of variantion, which will simplify the dataset used to run the KNN algorithm. 

This is just a extremely simplified  product recommendation example. There are still so much more topics needed to cover. For example, how do I deal with sparse matrix? how do I evaluate the model? and how do I predict the rating of other products?  



