from flask import Flask,request,render_template
import pickle
import numpy as np
drug_model = pickle.load(open('Random_for_drug_classification.pkl','rb'))
cardiovascular_model = pickle.load(open('cardiovuscular_disease_prediction_model.pkl','rb'))
heart_disease_model = pickle.load(open('heart_disease_prediction.pkl','rb'))
heart_failure_model = pickle.load(open('heart_failure_prediction.pkl','rb'))
diabetes_disease_model = pickle.load(open('diabetes_disease_prediction.pkl','rb'))
stroke_disease_model = pickle.load(open('stroke disease prediction.pkl','rb'))
app  = Flask(__name__)
@app.route('/')
def root():
    return render_template("home_page.html")
# @app.route('/index')
# def test():
#     return render_template("index.html")
@app.route('/drug_classification',methods=['POST','GET'])
def drug_classification():
    pr=""
    if request.method=='POST':
        values = [float(x) for x in request.form.values()]
        values= [np.array(values)]
        print(values)
        pr = drug_model.predict(values)
        print(pr)
    return render_template('predict.html',value=pr)
@app.route('/cardiovascular_predict',methods=['POST','GET'])
def cardiovascular_predict():
    pr=""
    if request.method=='POST':
        values = [float(x) for x in request.form.values()]
        values= [np.array(values)]
        print(values)
        pr = cardiovascular_model.predict(values)
        print(pr)
    return render_template('cardiovascular_page.html',value=pr)
@app.route('/heart_predict',methods=['POST','GET'])
def heart_predict():
    pr=""
    if request.method=='POST':
        values = [float(x) for x in request.form.values()]
        values= [np.array(values)]
        print(values)
        pr = heart_disease_model.predict(values)
        print(pr)
    return render_template('heart_disease_page.html',value=pr)

@app.route('/heart_failure_predict',methods=['POST','GET'])
def heart_failure_predict():
    pr=""
    if request.method=='POST':
        values = [float(x) for x in request.form.values()]
        values= [np.array(values)]
        print(values)
        pr = heart_failure_model.predict(values)
        print(pr)
    return render_template('heart_failure_page.html',value=pr)

@app.route('/diabetes_disease_predict',methods=['POST','GET'])
def diabetes_disease_predict():
    pr=""
    if request.method=='POST':
        values = [float(x) for x in request.form.values()]
        values= [np.array(values)]
        print(values)
        pr = diabetes_disease_model.predict(values)
        print(pr)
    return render_template('diabetes_disease_page.html',value=pr)
@app.route('/stroke_disease_predict',methods=['POST','GET'])
def stroke_disease_predict():
    pr=""
    if request.method=='POST':
        values = [float(x) for x in request.form.values()]
        values= [np.array(values)]
        print(values)
        pr = stroke_disease_model.predict(values)
        print(pr)
    return render_template('stroke_disease_page.html',value=pr)

@app.route('/about',methods=['POST','GET'])
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True,port='5000')