

import streamlit as st
from PIL import Image
import numpy as np
import os 
import tensorflow as tf
import tensorflow.keras as keras
from skimage.transform import resize
#caratula  


MODEL_PATH =  'D:/Proyecto_U3/Alexnet43_4.h5'


@st.cache(allow_output_mutation=True)
def cargar_modelo(MODEL_PATH):    
    model0 = keras.models.load_model(MODEL_PATH)
    return model0
model0 = cargar_modelo(MODEL_PATH)


width_shape = 192
height_shape = 192

def model_prediction(image, model):
    
    names = ['Ariel Chabla', 'ArroboMercy', 'BarraganSteven', 'CevallosJoan', 'EnriquezSelena', 'GenesisHeredia', 'GoyesAnthony', 'HectorCedeno', 'Jhon Zambrano', 'Jordan Espinosa', 'Jorge Borrero', 'JoseRuiz', 'LucioCarlos', 'MasacheFernando', 'MelanyLopez', 'MosqueraLucy', 'Nataly Acosta', 'OlallaLuis', 'ParragaMariaJose', 'PauteKevin', 'Raymond Davila', 'RivasSelena', 'SalazarJohana', 'Solano Wilmer', 'SolorzanoBryan', 'VinicioBorja']
    img_array = (tf.keras.preprocessing.image.img_to_array(image))
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) 
    value = names[np.argmax(score)]
    print(" {} ".format(names[np.argmax(score)]))
    return value

def main(): 
    
    menu = ["Portada","Clasificación"]#,"Añadir mas personas "]#,"Segmentacion","Dicomtopng"]
    choice = st.sidebar.selectbox("Menú",menu)
    
    if choice == "Portada":
        
       print('')
        
    elif choice == "Clasificación":
        caratula = Image.open( 'D:/Proyecto_U3/Logo_ESPE.png')
        st.image(caratula, width=700)
        
        
        st.header(' Reconocimiento  ')
        
        foto_2 = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])    #
        
        
        if foto_2 is not None:
            image2 = Image.open(foto_2)
            image2 = image2.resize([192,192],resample=Image.BICUBIC)
            image_2 = image2.convert("RGB")
            st.image(image2, caption="Imagen a Reconocer", use_column_width=False)
        #  predicción
        if st.button("Reconocer"):
            model0 =tf. keras.models.load_model(MODEL_PATH)
           
            impr = model_prediction(image2,  model0)
            st.success(f'{impr}')
            
 
    
    
if __name__ == '__main__':
    main()
