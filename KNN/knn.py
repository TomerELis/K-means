import numpy as np


class KNN:
    def __init__(self, k) -> None:
        self.saver = k  #save k

    def fit(self, x_train, y_train) -> None:
        self.vec, self.title  = x_train , y_train

    def predict(self, x_test):
        mylist = []
        for i in x_test:
            mylist.append(self.gotvec(np.array(i)))
        return np.array(mylist)

    def gotvec (self, newtitle):
        mylistvec = []
        for i in self.vec:
            mylistvec.append(np.sqrt(np.sum(np.square(np.array(i) - newtitle))))
        newme = np.argsort(mylistvec)[:self.saver]
        checky = []
        for k in range(self.saver):
            checky.append(self.title[newme[k]])
        bigger = checky[0]
        for i in checky:
            if checky.count(i) > checky.count(bigger):
                bigger = checky.count(i)
        return bigger





