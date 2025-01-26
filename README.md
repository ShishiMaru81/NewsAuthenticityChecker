# NewsAuthenticityChecker
A machine learning project to classify news articles as real or fake using natural language processing techniques.

Overview
The rapid spread of fake news poses significant challenges in today's digital age. This project aims to develop a machine learning model that can accurately classify news articles as fake or real based on their textual content.

Dataset
Source:Dataset imported from Kaggle.
Description: The dataset includes labeled news articles with two categories: FAKE and REAL by 1 and 0.And there are more 7 columns where title and author name also mentioned.
Size:The size of the dataset is almost (20000,8).[Dataset.shape]

Technologies Used
import numpy as np
import pandas as pd
import nltk
nltk.download("stopwords")
import sklearn
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

Project Workflow:
Data Cleaning
Data Preprocessing
Model Training
Evaluation by accuracy score testing

Thanks for checking my Code ,if there are any issue please comment there !
