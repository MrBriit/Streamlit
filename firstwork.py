

import streamlit as st 
import numpy as np 
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image


# #Set title

st.title('Total Data Science')

image = Image.open('tdslogo.png')
st.image(image,use_column_width=True)

#set subtitle

st.write("""
    # A simple Data App With Streamlit
    """)

st.write("""
 ### Let's Explore different classifiers and datasets
""")

# dataset_name = st.sidebar.selectbox(
#     'Select Dataset',
#     ('Breast Cancer','Iris', 'Wine')
# )

# #st.write(f"## {dataset_name} Dataset")

# classifier_name = st.sidebar.selectbox(
#     'Select classifier',
#     ('KNN', 'SVM')
# )

# def get_dataset(name):
#     data = None
#     if name == 'Iris':
#         data = datasets.load_iris()
#     elif name == 'Wine':
#         data = datasets.load_wine()
#     else:
#         data = datasets.load_breast_cancer()
#     X = data.data
#     y = data.target
#     return X, y

# X, y = get_dataset(dataset_name)
# st.dataframe(X)
# st.write('Shape of dataset:', X.shape)
# st.write('Number of classes in target:', len(np.unique(y)))


# fig = plt.figure()
# sns.boxplot(data=X, orient='h')

# st.pyplot()

# #plottinhg a histogram

# plt.hist(X)

# st.pyplot()


# #BUILDING OUR ALGORITHM

# #define a function necessary for our algorithms

# def add_parameter(name_of_clf):
#     params = dict()
#     if name_of_clf == 'SVM':
#         C = st.sidebar.slider('C', 0.01, 15.0)
#         params['C'] = C
#     else:
#         name_of_clf == 'KNN'
#         K = st.sidebar.slider('K', 1, 15)
#         params['K'] = K
#     return params

# #calling our function
# params = add_parameter(classifier_name)


# #accessing our classifier

# def get_classifier(name_of_clf, params):
#     clf = None
#     if name_of_clf == 'SVM':
#         clf = SVC(C=params['C'])
#     elif name_of_clf == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=params['K'])
#     else:
#         st.warning('select your choice of algorithm')
#     return clf


# clf = get_classifier(classifier_name, params)


# #Building our model

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# st.write(f'Classifier = {classifier_name}')
# st.write(f'Accuracy =', accuracy)

# #### PLOT DATASET ####
# # Project the data onto the 2 primary principal components
# pca = PCA(2)
# X_projected = pca.fit_transform(X)

# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

# fig = plt.figure()
# plt.scatter(x1, x2,
#         c=y, alpha=0.8,
#         cmap='viridis')

# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.colorbar()

# #plt.show()
# st.pyplot()





