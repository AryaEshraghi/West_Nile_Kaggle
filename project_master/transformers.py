import numpy as np
from os.path import basename
import pandas as pd
import re
import requests
import subprocess
import time

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

class DummyEncoder(TransformerMixin):
    """for one hot encoding a column from a df, in a pipeline, without having to split & featureunion"""

    def __init__(self, column):
        self.column = column # user specifies column to transform
        self.columns_ = list() # determined on fit
        self.dummies_ = pd.DataFrame() # determined on fit

    def transform(self, X, y=None, **kwargs):

        new_dummies = pd.get_dummies(X[self.column]) #get dummies of X
        aligned = self.dummies_.align(new_dummies,
                                      join='left',
                                      axis=1,
                                      fill_value=0) #check with fit dummies
        return pd.concat([X.drop(self.column,axis=1), aligned[1]], axis=1)

    def fit(self, X, y=None, **kwargs):
        self.dummies_ =  pd.get_dummies(X[self.column])
        self.columns_ = list(self.dummies_.columns)
        return self

class CVecTransformer(CountVectorizer):
    """
    turns CountVectorizer into a pipeline friendly transformer,
    while still allowing 1st level parameter access
    """

    def __init__(self,column=None,**kwargs):
        self.column = column
        super().__init__(**kwargs)

    def fit(self,X,y=None):
        super().fit_transform(X[self.column])
        return self

    def transform(self,X):
        matrix = super().transform(X[self.column])
        df = pd.DataFrame(matrix)
        return pd.concat([X, df], axis=1).drop(self.column)

    def fit_transform(self,X):
        return self.fit(X).transform(X)
    pass

class ColumnSelector(TransformerMixin):
    """docstring."""

    def __init__(self, columns=None, drop=False):
        """docstring"""
        self.columns = columns
        self.drop = drop

    def transform(self, X, y=None, **kwargs):
        """docstring"""
        if self.drop is False:
            return X[self.columns]
        else:
            return X.drop(self.columns,axis=1)

    def fit(self,X,y=None,**kwargs):
        return self


class ColumnMapper(TransformerMixin):
    """stores lambda func, target column for pd.Series.map(func) transforms"""

    def __init__(self,column=None,func=None,name=None,drop=False,**kwargs):
        """name == name of new column, column == column to map, func == lambda function to transform it """
        self.func = func
        self.name = name
        self.column = column
        self.drop = drop

    def fit(self,X,y=None,**kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        """map column, and drop original if True"""
        X[self.name] = X[self.column].map(self.func)
        if self.drop==True:
            return X.drop(self.column,axis=1)
        else:
            return X

class ColumnApplier(TransformerMixin):
    """stores lambda func, target column for pd.Series.apply(func) transforms"""

    def __init__(self,func=None,name=None,**kwargs):
        """name == name of new column, column == column to map, func == lambda function to transform it """
        self.kwargs = kwargs
        self.func = func
        self.name = name
        #self.column = column
        #self.drop = drop

    def transform(self, X, y=None,**kwargs):
        """map column, and drop original if True"""
        X[self.name] = X.apply(self.func,**self.kwargs)
        return X

    def fit(self,X,y=None,**kwargs):
        return self

class DfMerger(TransformerMixin):
    def __init__(self,df,**kwargs):
        self.df = df
        self.kwargs = kwargs
        return None
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return pd.merge(X,self.df,**self.kwargs)
