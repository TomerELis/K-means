# KNN
Constructor – The constructor receives a single argument named k, which dictates the
number of neighbors considered.


“fit” method
 The method receives a NumPy array of samples to be used as train data (called
x_train in the template file). In the array, each row is a data sample whose
dimensions equal the number of columns.
o The method receives a second 1-d NumPy array of labels per training sample to
be used as train data (called y_train in the template file).
o It doesn’t return anything.

“predict” method
The method receives a NumPy array of samples whose class is to be predicted. In
the array, each row is a data sample whose dimensions equal the number of
columns.
The method returns a 1-d NumPy array of labels, one per training sample.


# K-Means
Constructor – The constructor receives an argument named k, which dictates the
number of clusters considered, and max_iter which dictates the maximal number of
iterations done during the fit stage.

“initialize” method
The argument is a 2-d NumPy array. The number of rows is the same as the number of clusters, and each
row will represent the initial coordinates of some centroid.

“fit” method
The method receives a NumPy array of samples and uses it as train data (called
x_train in the template file). In the array, each row is a data sample whose
dimensions equal the number of columns. The method performs the k-means
algorithm on the train data to find centroids and train the model.
The method returns a dictionary. Its keys are clusters IDs that you assign during
training (unique integers of your choice). The values are 1-d NumPy arrays, each
of which represents a centroid.

“predict” method
The method receives a NumPy array of samples to be assigned to clusters. Each
row is a data sample whose dimensions equal the number of columns.
The method returns a 1-d NumPy array of cluster IDs. Each ID correlates to the
cluster to which the sample was assigned. Cluster IDs are the same ones
returned during training.

“wcss” method
The method will return the WCSS of the model, as calculated during
the “fit” stage.



