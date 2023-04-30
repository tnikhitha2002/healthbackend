from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

#Diabetes class
class diabetes_input(BaseModel):
    
    pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int       

#Kidney class
class kidney_input(BaseModel):
    
    sg : float
    htn : float
    hemo : float
    dm : float
    al : float
    appet : float
    rc : float
    pc : float      

#Heart class
class heart_input(BaseModel):
    
    age : int
    sex : int
    cp : int
    trestbps : int
    chol : int
    fbs  : int
    restecg  :  int
    thalach : int
    exang : int    
    oldpeak : float   
    slope   : int
    ca      : int
    thal    : int

# Cancer class
class cancer_input(BaseModel):
    
    texture_mean: float
    smoothness_mean : float
    compactness_mean : float
    symmetry_mean : float
    fractal_dimension_mean : float
    texture_se : float
    smoothness_se : float
    symmetry_se : float    
    symmetry_worst : float 

#Liver Class
class liver_input(BaseModel):
    
    Age    : int
    Gender : int
    Total_Bilirubin  : float
    Alkaline_Phosphotase : float
    Alamine_Aminotransferase : int
    Aspartate_Aminotransferase : int
    Total_Protiens  : float
    Albumin   : float   
    Albumin_and_Globulin_Ratio : float   
        

# loading the saved model of diabetes
diabetes_model = pickle.load(open('./diabetes_model.sav', 'rb'))

# loading the saved model of kidney
kidney_model = pickle.load(open('./kidney_model.sav', 'rb'))

# loading the saved model of heart
heart_model = pickle.load(open('./heart_model.sav', 'rb'))

# loading the saved model of cancer
cancer_model = pickle.load(open('./cancer_model.sav', 'rb'))

# loading the saved model of liver
liver_model = pickle.load(open('./liver_model.sav', 'rb'))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():    
    return "Hello World!"

#Diabetes Prediction
@app.post('/diabetes_prediction')
def diabetes_predd(input_parameters : diabetes_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

 #Kidney prediction   
@app.post('/kidney_prediction')
def kidney_predd(input_parameters : kidney_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    spGra = input_dictionary['sg']
    hyp = input_dictionary['htn']
    hmgb = input_dictionary['hemo']
    dia = input_dictionary['dm']
    alb = input_dictionary['al']
    app = input_dictionary['appet']
    rbc = input_dictionary['rc']
    pus = input_dictionary['pc']
    
    
    input_list = [spGra,hyp,hmgb,dia,alb,app,rbc,pus]
    
    prediction = kidney_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'The person is not at the risk of getting a kidney disease'
    else:
        return 'The person is at the risk of getting a kidney disease'
    
#heart Prediction
@app.post('/heart_prediction')
def heart_predd(input_parameters : heart_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    ag = input_dictionary['age']
    sx = input_dictionary['sex']
    cpp = input_dictionary['cp']
    tre = input_dictionary['trestbps']
    cho = input_dictionary['chol']
    fb = input_dictionary['fbs']
    rest = input_dictionary['restecg']
    thal = input_dictionary['thalach']
    exa = input_dictionary['exang']
    old = input_dictionary['oldpeak']
    slp = input_dictionary['slope']
    caa = input_dictionary['ca']
    thl = input_dictionary['thal']
    
    
    
    input_list = [ag,sx,cpp,tre,cho,fb,rest,thal,exa,old,slp,caa,thl]
    
    prediction = heart_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'The person is not at the risk of getting a heart disease'
    else:
        return 'The person is at the risk of getting a heart disease'
    
@app.post('/cancer_prediction')
def diabetes_predd(input_parameters : cancer_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    tm = input_dictionary['texture_mean']
    sm = input_dictionary['smoothness_mean']
    cm = input_dictionary['compactness_mean']
    sym = input_dictionary['symmetry_mean']
    fdm = input_dictionary['fractal_dimension_mean']
    ts = input_dictionary['texture_se']
    smse = input_dictionary['smoothness_se']
    syse = input_dictionary['symmetry_se']
    syw = input_dictionary['symmetry_worst']
    
    
    input_list = [tm,sm,cm,sym,fdm,ts,smse,syse,syw]
    
    prediction = cancer_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'The person is not at the risk of getting a breast cancer'
    else:
        return 'The person is at the risk of getting a breast cancer'
    
#Liver prediction
@app.post('/liver_prediction')
def liver_predd(input_parameters : liver_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    age = input_dictionary['Age']
    gend = input_dictionary['Gender']
    bili = input_dictionary['Total_Bilirubin']
    alk = input_dictionary['Alkaline_Phosphotase']
    ala = input_dictionary['Alamine_Aminotransferase']
    aspr = input_dictionary['Aspartate_Aminotransferase']
    protiens = input_dictionary['Total_Protiens']
    alb = input_dictionary['Albumin']
    albRatio = input_dictionary['Albumin_and_Globulin_Ratio']
    
    
    
    input_list = [age,gend,bili,alk,ala,aspr,protiens,alb,albRatio]
    
    prediction = liver_model.predict([input_list])
    
    if (prediction[0] == 2):
        return 'The person is not at the risk of getting a liver disease'
    else:
        return 'The person is at the risk of getting a liver disease'