import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as mobilenet_v2_preprocess_input



st.header("Indentifikasi Varietas Mangga Berdasarkan Citra Daun Menggunakan CNN ")
model = tf.keras.models.load_model("/content/gdrive/MyDrive/ColabNotebooks/model/ModelVGG16epoch20.h5")

### load file
uploaded_file = st.file_uploader("Pilih file gambar", type="jpg")

map_dict = {0: 'Mangga Apel',
            1: 'Mangga Gedong',
            2: 'Mangga Golek',
            3: 'Mangga Lalijiwo',
            4: 'Mangga Manalagi',
            5: 'Mangga Wirasangka',}
            
# cm=confusion_matrix(df_test.Label,pred)
#     clr = classification_report(df_test.Label, pred, target_names=["daun_mangga_apel","daun_mangga_gedong","daun_mangga_golek","daun_mangga_lalijiwo","daun_mangga_manalagi","daun_mangga_wirasangka"])
#     print(clr)
#     # Display 6 picture of the dataset with their labels
#     fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 15),
#                         subplot_kw={'xticks': [], 'yticks': []})

#     for i, ax in enumerate(axes.flat):
#         ax.imshow(plt.imread(df_test.File_Path.iloc[i+1]))
#         ax.set_title(f"True: {df_test.Label.iloc[i+1]}\nPredicted: {pred[i+1]}")
#     plt.tight_layout()
#     plt.show()


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # prediction = model.predict(img).flatten()
    # return {labels[i]:float(prediction[i]) for i in range(6)}

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Klik Untuk Identifikasi")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Varietas {}".format(map_dict [prediction]))