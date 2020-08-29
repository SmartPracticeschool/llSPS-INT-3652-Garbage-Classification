from flask  import Flask  ,render_template , request
import os
from keras.preprocessing import image 
from werkzeug.utils import  secure_filename 
from keras.models import load_model 
model = load_model("garbage.h5")

import numpy as np
app = Flask(__name__)

 

import tensorflow as tf
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
tf.compat.v1.keras.backend.set_session(sess)






@app.route('/')
def index():
    return render_template("base.html" , methods = ['GET'])
 
@app.route('/predict',methods = ['GET','POST'])
def pred():
    if request.method == "POST":
        print("hi")
        f = request.files["image"]
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(basepath,"uploads",secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        img =image.load_img(file_path,target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
         
        with sess.as_default():
            p=model.predict_classes(x)
            print(p)
        index =['glass', 'metal', 'paper', 'plastic']
        text ="the prediction is:" +index[p[0]]
        return text
        
if __name__=='__main__':
    app.run(debug = False)
    
    
    