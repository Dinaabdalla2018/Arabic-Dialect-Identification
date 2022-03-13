from flask import Flask,request, url_for, redirect, render_template
import numpy as np
from keras.models import *
from sklearn.feature_extraction.text import CountVectorizer
import pickle

app = Flask(__name__)  
@app.route('/')
def index():
  return render_template('index.html')

def Load__model(name_file):
  # Load Model from a file
  model = pickle.load(open(name_file, 'rb'))
  return model 

@app.route('/Predict',methods=['POST','GET'])
def predict():
	model__ = request.form.get("model")
	class_label = ['EG','PL','KW','LY','QA','JO','LB','SA','AE','BH','OM','DZ','SY','IQ','SD','MA','YE','TN']
	if (str(model__) == 'SVM'):
		vocabulary_   = Load__model('Model/featurem.pkl')
		loaded_vec    = CountVectorizer(decode_error=" Replace ", vocabulary=vocabulary_)
		loaded_tf_idf = Load__model('Model/tfidftransformerm.pkl')
		test_tfidf    = loaded_tf_idf.transform(loaded_vec.transform([str(request.form.get("name_input"))])).toarray()
		model         = pickle.load(open('Model/svm.sav', 'rb'))
		pred          = model.predict(test_tfidf)[0]
		return render_template("index.html", prediction='Your words belong to the dialect '+ pred)

	elif (str(model__) == 'DL'):
		vocabulary_   = Load__model('Model/feature.pkl')
		loaded_vec    = CountVectorizer(decode_error=" Replace ", vocabulary=vocabulary_)
		loaded_tf_idf = Load__model('Model/tfidftransformer.pkl')
		test_tfidf    = loaded_tf_idf.transform(loaded_vec.transform([str(request.form.get("name_input"))])).toarray()
		model         = load_model('Model/my_model.h1')
		pred          = model.predict(test_tfidf)
		output        = class_label[np.argmax(pred)]	
		return render_template("index.html", prediction='Your words belong to the dialect '+ output)
 
if __name__ == '__main__':
    app.run(debug=True)