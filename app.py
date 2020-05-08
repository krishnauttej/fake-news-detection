from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
from sklearn.model_selection import train_test_split

# load the model from disk
filename = 'Fake.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('vector.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
# import re
# import nltk
# nltk.download('stopwords')
# import pandas as pd
# df=pd.read_csv('train.csv')
# import pickle
# 
# 
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# #from nltk.stem import WordNetLemmatizer
# ps = PorterStemmer()
# #word=WordNetLemmatizer()
# 
# kk = []
# for i in range(len(df)):
#     review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
#     review = review.lower()
#     review = review.split()
#     review = [ps.stem(word) for word in review if not word in (stopwords.words('english'))]
#     review = ' '.join(review)
#     kk.append(review)
#     
#     
#     
# 
# from sklearn.feature_extraction.text import TfidfVectorizer
# cv = TfidfVectorizer(max_features=5000)
# x = cv.fit_transform(kk).toarray()
# pickle.dump(cv, open('vector.pkl', 'wb'))
# 
# 
# Y=df['label']
# # Train Test Split
# 
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size = 0.20, random_state = 0)
# 
# # Training model using Naive bayes classifier
# 
# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB()
# model.fit(X_train, y_train)
# model.score(X_test,y_test)
# pickle.dump(model, open('Fake.pkl', 'wb'))
# 
# y_pred=model.predict(X_test)
# 
# from sklearn.metrics import confusion_matrix,accuracy_score
# con=confusion_matrix(y_test,y_pred) 
# acc=accuracy_score(y_test,y_pred)


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)