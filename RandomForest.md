
# Random Forest
Just as a forest is a collection of trees, a Random forest algorithm consists of a group of decision trees. The way a decision tree learns patterns from data is similar to humans thinking process involved in solving a complex problem. In other words, humans can fathom a complex problem by asking a sequence of simple yet interpretable questions. Likewise, a decision tree can understand the structure of data by creating a series of explainable logical questions, which in turn makes them transparent models.


The mathematical details of how these questions are asked to classify a data point can be explained using Gini impurity.

Consider a simple dataset with two features x1 and x2 with each data point belonging to either class1 or class2
 ![Simple dataset](https://i.imgur.com/NaCL95Z.png)
where * represents the data points that belong to class1 and O for class 2.

Clearly, data cannot be separated by a linear model, however, since the decision tree is a non-linear model it can separate the data points by creating multiple linear boundaries or decision planes as shown above.

**Visualizing decision tree**
![RF_tree](https://i.imgur.com/cTPQ87Z.png)
where
 - X[1] - represents **x2**
 - X[0] - represents **x1**
 - gini - is a impurity metric helps the algorithm to decide which is the best feature split while creating the nodes in tree.
 - samples - number of data points
 - value = indicates frequency of class1 and class2 data points 

The tree above elucidates how a decision tree interprets the data with the hierarchical rational questions on features. Every box in the tree is called a node, which symbolizes a query on a single feature with binary outcome True or False. Furthermore, the first node of a tree is called root node, the last nodes are called leaf nodes and the depth of the tree is defined as the number of levels in the tree(in our tree it is 4).

**Understand Gini Impurity**
Intuitively Gini impurity gives the randomness in the sample data at any given node in a tree. Which means the model tries to reduce the randomness in data as it creates splits via nodes in the tree. Thereby, it can predict the right class of a data point with a higher probability.

To comprehend this, consider the decision tree above, at the root node with the Gini impurity is 0.497 with 13 samples out of which 7 belong to class1 and 6 to class2.

So, the probability of classifying a data point as class1 at this node is 7/13 = 0.538.

Now consider the last node with 1 sample for which the class1 prediction probability would be 1/1 = 1.

Mathematically Gini impurity can be defined as 
$$
I_{G}(n) = 1 - \sum_{i=1}^{k}(p_{i})^2
$$
where, 

 - n = node
 - k = number of classes, in our case its 2
 - pi = is the probability of selecting a data point from class i at node n. Which is the ratio of number of data points in class i to total number of points at node n.

And this equation is trying to capturing the probability of any randomly selected sample data point being wrongly classified if the classification is done using the distribution of the sample in the node.



**Calculating Gini impurity:**
At root node
$$
I_{root} = 1 - \left ( \left ( \frac{7}{13}  \right )^2  + \left ( \frac{6}{13}  \right )^2 \right ) = 0.497
$$

At node x <=0.45
$$
I_{left} = 1 - \left ( \left ( \frac{2}{6}  \right )^2  + \left ( \frac{4}{6}  \right )^2 \right ) = 0.444
$$

At node x <= 0.575
$$
I_{right} = 1 - \left ( \left ( \frac{5}{7}  \right )^2  + \left ( \frac{2}{7}  \right )^2 \right ) = 0.408
$$

So, overall gini impurity for second layer 
$$
I_{secondLayer} = \frac{n_{left}}{n_{root}} * I_{left} + \frac{n_{right}}{n_{root}} * I_{root} = 0.423 
$$

$$
I_{secondLayer} < I_{root} 
$$

Clearly, overall Gini impurity is decreasing from root to leaf nodes.

**Variance-Bias tradeoff**
Grasping the idea of variance-bias tradeoff in decision trees helps one to apprehend the need for Random Forest model.

High variance is a situation where the model tries to memorize the patterns in the data rather than learning the underlying general pattern. And in case of high bias, the model doesn't have enough capacity to understand the complex patterns in data. It is like teaching a first-grade student about calculus which beyond the student's comprehension ability.

In the case of decision trees, the knowledge learned from the data is controlled by the depth of a tree. If a tree has an inadequate depth then it suffers from high bias because the model cannot ask enough questions to make sense of all the patterns in the complex data. However, if the tree has a higher depth then it tries to perfectly fit for training data by memorizing it, which makes the model weaker to generalize for real data.

Which means an ideal decision tree should maintain a low bias(to have enough potential to learn complex patterns in data) and low variance(to learn from a general pattern in data rather than memorizing patterns).

**Random Forests as a solution**
As explained earlier Random Forest is a collection of decision trees. However, this model doesn't give the final prediction by simply averaging the predictions of each decision tree.
The word 'random' is used in Random forests because,

 1. It builds each decision tree on a random sample drawn from the data with replacement.
 2. Random subsets of features are used for splitting nodes of each tree.

This method of sampling random sets from data with replacement is called bootstrapping. And the reason behind using it is that, if each tree is trained on a random sample of data with maximum depth then each tree develops high variance with respect to that sample. However, the overall variance of the collection of trees will be low without increasing bias. Additionally, this method of training different decision trees on random samples from data and averaging the final predictions is called bagging.

If all trees are trained on random samples from data with all features then multiple trees can have the same kind of feature splits. Which creates redundant trees because most of them will be duplicates. Hence, Random forest uses a random subset of features for splitting nodes in every tree. 

This random subset size is generally calculated as square_root(k), where k is the number of features. So, if k=4, then every node split in each tree will get 2 random features.
