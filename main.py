import time
import os
import cv2
import numpy as np
import tensorflow       
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('/select.html', )

@app.route("/compare")
def compare_template():
    return render_template('/compare.html', )    

@app.route('/predicts', methods=['POST'])
def predicts():
    chosen_model = request.form['select_model']
    model_dict = { 'CNN Model Skenario 1':'static/MLModule/skenario1.h5',
                   'CNN Model Skenario 2':'static/MLModule/skenario2.h5',
                   'CNN Model Skenario 3':'static/MLModule/skenario3.h5',
                   'CNN Model Skenario 4':'static/MLModule/skenario4.h5' }
    if chosen_model in model_dict:
        model = load_model(model_dict[chosen_model]) 
    elif chosen_model in model_dict:
        model = load_model(model_dict[1])
    else:
        model = load_model(model_dict[2])

    filename = request.form.get('input_image')
    img = cv2.cvtColor(np.array(Image.open(filename)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    calculation = (pred > 0.5).astype(np.int)
    print(calculation)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return predict_result(chosen_model, runtimes, respon_model, calculation , filename[7:])

@app.route('/predict', methods=['POST'])
def predict():
    chosen_model = request.form['select_model']
    model_dict = { 'CNN Model Skenario 1':'static/MLModule/skenario1.h5',
                   'CNN Model Skenario 2':'static/MLModule/skenario2.h5',
                   'CNN Model Skenario 3':'static/MLModule/skenario3.h5',
                   'CNN Model Skenario 4':'static/MLModule/skenario4.h5' }
    if chosen_model in model_dict:
        model = load_model(model_dict[chosen_model]) 
    elif chosen_model in model_dict:
        model = load_model(model_dict[1])
    else:
        model = load_model(model_dict[2])

    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    calculation = (pred > 0.5).astype(np.int)
    print(calculation)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return predict_result(chosen_model, runtimes, respon_model, calculation , 'temp.jpg')

def predict_result(model, run_time, probs, result, img):
    class_list = {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}
    idx_pred = probs.index(max(probs))
    vector = np.vectorize(np.int)
    result1 = vector(result)
    labels = list(class_list.keys())
    return render_template('/result_select.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, result2 = result1, 
                            run_time=run_time, img=img)


#COMPARE
@app.route('/predict_compare', methods=['POST'])
def predict_compare():
    respon_model = []
    running_time = []
    chosen_model = request.form.getlist('select_model')
    filename = request.form.get('input_image')
    img = cv2.cvtColor(np.array(np.array(Image.open(filename))), cv2.COLOR_BGR2RGB)
    model_dict = {'CNN Model Skenario 1' :'static/MLModule/skenario1.h5',
                   'CNN Model Skenario 2':'static/MLModule/skenario2.h5',
                   'CNN Model Skenario 3':'static/MLModule/skenario3.h5',
                   'CNN Model Skenario 4':'static/MLModule/skenario4.h5'}

    for m in chosen_model:
        model = load_model(model_dict[m])
        imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        start = time.time()
        pred = model.predict(imgs)[0]
        running_time.append(round(time.time()-start,4)) 
        respon_model.append([round(elem * 100, 2) for elem in pred]) 
    return result_compare(respon_model, chosen_model, running_time, filename[7:])

@app.route('/predicts_compare', methods=['POST'])
def predicts_compare():
    respon_model = []
    running_time = []
    chosen_model = request.form.getlist('select_model')
    filename = request.files['file']
    img = cv2.cvtColor(np.array(np.array(Image.open(filename))), cv2.COLOR_BGR2RGB)
    model_dict = {'CNN Model Skenario 1':'static/MLModule/skenario1.h5',
                   'CNN Model Skenario 2':'static/MLModule/skenario2.h5',
                   'CNN Model Skenario 3':'static/MLModule/skenario3.h5',
                   'CNN Model Skenario 4':'static/MLModule/skenario4.h5'}

    for m in chosen_model:
        model = load_model(model_dict[m])
        imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        start = time.time()
        pred = model.predict(imgs)[0]
        running_time.append(round(time.time()-start,4)) 
        respon_model.append([round(elem * 100, 2) for elem in pred]) 
    return result_compare(respon_model, chosen_model, running_time, 'temp.jpg')               



def result_compare(probs, mdl, run_time, img):
    class_list = {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2} 
    idx_pred = [i.index(max(i)) for i in probs]
    labels = list(class_list.keys())
    return render_template('/result_compare.html', labels=labels, 
                            probs=probs, mdl=mdl, run_time=run_time, pred=idx_pred, img=img)    

if __name__ == "__main__": 
        app.run(debug=True, host='0.0.0.0', port=2000)