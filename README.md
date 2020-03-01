### Data Analytics CS381 - Predicting Test Scores

#### Group project analyzing a set of student data using a supervised machine learning model called Support-Vector Machine to what factors, if any, contribute to student test scores. 


Data Understanding 

The dataset for this project consists of 1,000 instances of anonymous student data. Each instance includes the attributes of Gender, Race/Ethnicity, Parental level of education, Lunch fee, and Test prep courses taken, if any. The instances also include a score for Math, Reading, and Writing. Math Scores were selected as the target variable for the model, and the other attributes are independent variables to help predict the target variable. Some of these independent variables are non-binary, Parental level of education, for example. This variable has a several different classes, including: some high school, high school, some college, associate’s degree, bachelor’s degree, and master’s degree. We used a classification model to determine which attributes influence the student’s math scores the most. This gives us good information to present to schools about what factors can be looked at to improve test scores.

Data Preparation 

To prep the data for analysis, we used python and python libraries like pandas to automate the task of converting the text/string data into numeric representation. The data preparation implementation is done in the file preprocess.ipynb.

From the start, our data was mostly represented in string format which is not ideal for data mining. Below are the first 10 rows in its original format:



Step one was taking the median score for each test subject so that we could set a benchmark to compare against. Students with test scores above the average were represented with a ‘1’ and students below the average were represented with a ‘0’. Our median results were as follows:

Median reading score: 70.0
Median writing score: 69.0
Median math score: 66.0


Step two consisted of converting Gender, Race/Ethnicity, Lunch, and Test Preparation Course into integer form to mine the data. These attributes were converted as follows:

gender:
 {'female': 0, 'male': 1}
 
race/ethnicity:
 {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}
 
lunch:
 {'free/reduced': 0, 'standard': 1}
 
test preparation course:
 {'completed': 0, 'none': 1}
 
Step three was to map the parental level of education to number representation. Parental education was mapped as:

education_map = {'some high school': 0,
             	    "high school": 1,
             	    'some college': 2,
             	    "associate's degree": 3,
             	    "bachelor's degree": 4,
       	    "master's degree": 5,}
 
Finally we get the numeric representation we need model the data.

 


Model

The model we used is Support-Vector Machine. A Support-vector Machine or SVMs are supervised machine learning models. SVMs work by constructing an n-1 dimensional hyperplane or set of hyperplanes for n dimensional space, which can be used for classification.

Initially we used decision trees, however they had low training error, but high test errors, meaning overfitting. SVMs on the other hand had a slightly higher training error but low test error, which means that SVMs were not overfitting the data as much as decision trees.
Predicting which students are going to do poorly due to their specific circumstances such as parent education or free vs paid lunches.

The python code below uses an SVM model from Scikit-learn library. The highlighted line splits the dataset into the training data and the test data. The test size is one third the size of the dataset.

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

if __name__=='__main__':
    df = pd.read_csv('dataset/processed.csv')
    df = df.drop(['reading score', 'writing score'],axis=1)
 
    X = df.drop(['math score'], axis=1)
    y = df['math score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)


    model = SVC(gamma='auto' ,kernel='rbf')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

Evaluation 

Model was able to predict math score category when given reading scores with ~80% accuracy. 
The model was only able to predict math score category with ~68% accuracy when reading scores were not given.

Based on the results we can say the free lunch, test preparation etc. have only a small correlation in math scores compared to the student’s reading score. 
   


