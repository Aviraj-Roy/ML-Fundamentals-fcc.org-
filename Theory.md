__**DAY 1**__

AI - The effort to automate intellectual tasks normally peformed by humans. Predefined set of rules.
Computer programs that play tic-tac-toe or chess against human players are examples of simple artificial intelligence.
Artificial Intelligence (Machine Learning (Neural Networks))

Machine Learning - Figures out the rules for us. Generates rules based on the given inputs & outputs. It is a subset of Artificial Intelligence.

Neural Network - A form of machine learning that uses a layered representation of data. It is made up of interconnected nodes. Neural networks aren't modeled after the way the human brain works.

Features -> Input to the model
Label -> Output to the model

Tensor - vector(a data point) generalized to a higher dimension.
A tensor is a generalization  of vectors and matrices to n-dimensional structures. Internally, TensorFlow represents tensors as n-dimensional arrays of vase datatypes.
Each tensor represents a partially defined computation that will eventually produce a value.
Each tensor has a data type and a shape.
Data Types:- float32, int32, strings and others.
Shape:- Represent the dimension of data.
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float16)
Above are scalar and therefore have 1 value.

Rank/Degree - no. of dimensions
rank1_tensor = tf.Variable(["Test", "Ok", "Tim"], tf.string)
rank2_tensor = tf.Variable([["test", "this", "ok"], ["test", "yes", "success" ]], tf.string)
Call method -> tf.rank(rank2_tensor) -> Determine rank

Shape of Tensors -> How many items in each dimension
Method -> rank2_tensor.shape -> O/P= [2,3]

Changing Shape
Code
tensor1 = tf.ones([1,2,3]) #tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1]) #reshapes existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3,-1]) #-1 tells the tensor to calculate the size of the dimension in that place. this will reshape the tensor to [3,2]
#the no. of elements in the reshaped tensor MYST match the no. in the original 

Types of TENSORS
    Variable (mutable)
    Constant (immutable)
    Placeholder (immutable)
    SparseTensor (immutable)

Evaluating Tensors
#Default template code
with tf.Session() as sess: #creates a session using the defult graph
    tensor.eval() #tensor will of course be the name of your tensor

%tensorflow_version 2.x
import tensorflow as tf
print(tf.version)

t = tf.zeros([5,5,5,5])
print(t)
t = tf.reshape(t, [625])
print(t)


__**DAY 2**__

TensorFlow Core Learning Algorithms
1. Linear Regression (predict numeric values)
    Line of Best Fit - refers to a line through a scatter plot of data points that best expresses the relationship between those points.
2. Classification
3. Clustering
4. Hidden Markov Models

Data (Titanic Data Set) - has tons of information about each passenger on the ship.
#Code of Load Dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

the "pd.read_csv()" method will return to us a new pandas dataframe.
To Find one specific dataframe, use <database_name>.loc[i]
What does the panda.head() function do?
Shows us the first 5 entries(rows)


__**DAY 3**__

Training DataSet - feed to the model so that it can develop and learn. Usually much larger sizr than the testing data.
Testing DataSet - used to evaluate the model and see the performance. P.S:- Different dataset to be used.
Feature Columns
    1. Categorical Data
    2. Numeric Data


__**DAY 4**__

Small batches of 32.
Feed the data points from dataset according to the number of epochs.
Epoch - One stream of our entire dataset.
No. of epochs we define is the amount of times our model will see the entire dataset. Use multiple epochs  for more accuracy but it'll take longer time.
Overfeeding - Pass too much training data, memorizes it. Horrible for testing data.
Input Function -  A way to feed in the data into TF, how to break into epochs


__**DAY 5**__
Generally, model data is streamed in small batches of 32.

