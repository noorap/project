from flask import Flask,render_template,request
import pickle

app=Flask(__name__)
model=pickle.load(open('projmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    import pdb; pdb.set_trace()
    sensor_00=float(request.form['Sensor 00'])
    sensor_01=float(request.form['Sensor 01'])
    sensor_02=float(request.form['Sensor 02'])
    sensor_03=float(request.form['Sensor 03'])
    sensor_04=float(request.form['Sensor 04'])
    sensor_05=float(request.form['Sensor 05'])
    sensor_06=float(request.form['Sensor 06'])
    sensor_07=float(request.form['Sensor 07'])
    sensor_08=float(request.form['Sensor 08'])
    sensor_09=float(request.form['Sensor 09'])
    sensor_10=float(request.form['Sensor 10'])
    sensor_11=float(request.form['Sensor 11'])
    sensor_12=float(request.form['Sensor 12'])
    sensor_13=float(request.form['Sensor 13'])
    sensor_14=float(request.form['Sensor 14'])
    sensor_15=float(request.form['Sensor 15'])
    sensor_16=float(request.form['Sensor 16'])
    sensor_17=float(request.form['Sensor 17'])
    sensor_18=float(request.form['Sensor 18'])
    sensor_19=float(request.form['Sensor 19'])
    sensor_20=float(request.form['Sensor 20'])
    sensor_21=float(request.form['Sensor 21'])
    sensor_22=float(request.form['Sensor 22'])
    sensor_23=float(request.form['Sensor 23'])
    sensor_24=float(request.form['Sensor 24'])
    sensor_25=float(request.form['Sensor 25'])
    sensor_26=float(request.form['Sensor 26'])
    sensor_27=float(request.form['Sensor 27'])
    sensor_28=float(request.form['Sensor 28'])
    sensor_29=float(request.form['Sensor 29'])
    sensor_30=float(request.form['Sensor 30'])
    sensor_31=float(request.form['Sensor 31'])
    sensor_32=float(request.form['Sensor 32'])
    sensor_33=float(request.form['Sensor 33'])
    sensor_34=float(request.form['Sensor 34'])
    sensor_35=float(request.form['Sensor 35'])
    sensor_36=float(request.form['Sensor 36'])
    sensor_37=float(request.form['Sensor 37'])
    sensor_38=float(request.form['Sensor 38'])
    sensor_39=float(request.form['Sensor 39'])
    sensor_40=float(request.form['Sensor 40'])
    sensor_41=float(request.form['Sensor 41'])
    sensor_42=float(request.form['Sensor 42'])
    sensor_43=float(request.form['Sensor 43'])
    sensor_44=float(request.form['Sensor 44'])
    sensor_45=float(request.form['Sensor 45'])
    sensor_46=float(request.form['Sensor 46'])
    sensor_47=float(request.form['Sensor 47'])
    sensor_48=float(request.form['Sensor 48'])
    sensor_49=float(request.form['Sensor 49'])
    sensor_50=float(request.form['Sensor 50'])
    sensor_51=float(request.form['Sensor 51'])

   
    output=model.predict([[sensor_00,sensor_01,sensor_02,sensor_03,sensor_04,sensor_05,sensor_06,sensor_07,sensor_08,sensor_09,sensor_10,sensor_11,sensor_12,sensor_13,sensor_14,sensor_15,sensor_16,sensor_17,sensor_18,sensor_19,sensor_20,sensor_21,sensor_22,sensor_23,sensor_24,sensor_25,sensor_26,sensor_27,sensor_28,sensor_29,sensor_30,sensor_31,sensor_32,sensor_33,sensor_34,sensor_35,sensor_36,sensor_37,sensor_38,sensor_39,sensor_40,sensor_41,sensor_42,sensor_43,sensor_44,sensor_45,sensor_46,sensor_47,sensor_48,sensor_49,sensor_50,sensor_51]])[0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    return render_template('result.html',predicton_text='Pump Sensor is {}'.format(output))
if __name__ == '__main__':
    app.run(port=5000)
        