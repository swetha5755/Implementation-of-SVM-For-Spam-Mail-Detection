# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries.
2. Use chardet to detect the encoding of the file spam.csv.
3. Load the CSV file using the encoding (in this case, 'windows-1252').
4. Display the first few rows and general information.
5. Ensure the dataset does not contain null or missing values.
6. Feature (x) = text messages (column "v2").
7. Label (y) = spam/ham classification (column "v1").
8. Split the Data into Training and Testing Sets.
9. Use CountVectorizer to convert text data into numerical vectors.
10. Use Support Vector Classifier to train on the vectorized data.
11. Predict whether messages are spam or not.
12.Use accuracy score to evaluate the model performance. 


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Swetha S
RegisterNumber:  212224040344
import chardet
file='spam.csv'
print("Name: Swetha S\nReg.no: 212224040344")
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
*/
```

## Output:
<img width="762" height="94" alt="image" src="https://github.com/user-attachments/assets/6cbb8ab5-65a6-473b-be52-af1b5541f1e6" />

<img width="850" height="222" alt="image" src="https://github.com/user-attachments/assets/a30414f3-2357-4489-8194-6877706bdbed" />

<img width="446" height="261" alt="image" src="https://github.com/user-attachments/assets/ee203a25-f913-4177-a36e-fce521d4fbcf" />

<img width="320" height="142" alt="image" src="https://github.com/user-attachments/assets/e191561b-f932-460d-b092-7efbf7c98e55" />

<img width="768" height="34" alt="image" src="https://github.com/user-attachments/assets/aebef70f-7de4-45bc-a96e-aaa8fc30b8a5" />

<img width="276" height="40" alt="image" src="https://github.com/user-attachments/assets/28ed5f33-b624-461c-b00e-5a8127937769" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
