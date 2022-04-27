# Import of the libraries used in this app

import os, random
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps 
import tensorflow as tf
import time

# Delete the streamlit footer
hide_footer = """
        <style>
        footer {visibility : hidden;}
        </style>
        """
st.markdown(hide_footer, unsafe_allow_html=True)

# Create the sections of the streamlit

header = st.container()
dataset = st.container()
data_analysis = st.container()
test_images_display = st.container()

# Add the sidebar

navigation = st.sidebar.radio("Let's do some Mushroom Recognition !", ["Introduction","Dataset", "Data analysis", "Predictions with deep learning"])

# Final dataframe used :

df = pd.read_csv("df_alldatas_nettoye.csv")

# Fill-in each section 

if navigation == "Introduction":
    with header:
        st.title("Let's do some Mushroom Recognition !")
        st.header("Introduction")
        st.write("The existence of a large literature on the characteristics of fungi allows,",
                 " once the fungus is identified, inform the user about the toxicity and ",
                 "other parameters of the mushroom. It is important to identify the fungus at the outset.",
                 " Therefore, this application will allow you to identify the mushroom species of your choice.")
        img_mushrooms = Image.open('img all.png')
        st.image(img_mushrooms, caption='Edible? or poisonous?')

if navigation == "Dataset":
    with dataset:
        #st.title("Let's do some Mushroom Recognition !")
        st.header('Mushroom observer dataset')
        st.write("The observations used in this project were collected from the GitHub repository, ",
                 "built with data from http://mushroomobserver.org. This site has been around since 2006,",
                 "  data is collected thanks to mushroom enthusiasts who contribute daily to the collection ",
                 "of mushroom observations on the site. There were then about 10,100 users who  contributed",
                 " a total of about 446,863 mushroom sightings. Each observation contained one to five images.")
        
        
        df_initial = pd.read_csv("dataset_initial.csv")
        st.dataframe(df_initial.head(), 1000, 1000)
        st.write("The size of the final dataset is :", df_initial.shape)
        
        st.subheader("Data cleaning")
        
        st.write("After removing missing data and filtering variables, only 3 variables were considered relevant for this study and kept.")

        df = pd.read_csv("df_alldatas_nettoye.csv")
        st.dataframe(df.head())
        st.write("The size of the final dataset is", df.shape)
        
        
if navigation == "Data analysis":
    with data_analysis:
        #st.title("Let's do some Mushroom Recognition !")
        st.header('Mushroom observer dataset')
        st.subheader("Data distribution")
        
        options = ["Familly", "Specie"]
        
        choix_dist = st.selectbox("Mushroom distrubtion by :", options = options)
        
        if choix_dist == options[0]:
            family_dist = pd.DataFrame(df["family"].value_counts()).head(15)
            st.bar_chart(family_dist)
            
        else :
            species_dist = pd.DataFrame(df["species"].value_counts()).head(15)
            st.bar_chart(species_dist)

        st.subheader("How many species of mushrooms per familly ? ")
        
        list_species = []
        family_freq = df["family"].value_counts().index.tolist()
        for family in family_freq :
            df_chart = df[df['family'] == family]
            list_species.append(df_chart["species"].value_counts().shape[0])
                
        #Data Set
        df_chart_plot = pd.DataFrame(list(zip(family_freq, list_species)), columns = ["Familly", "Specie"])

        #The plot
        fig = px.line(
            df_chart_plot, 
            x = "Familly", 
            y = "Specie",
            title = "Species per familly"
            )
        fig.update_traces(line_color = "blue")
        fig.update_layout(title_text="Total species per familly", title_x=0.5)
        st.plotly_chart(fig)
        
        st.subheader('Familly or specie ?')
        st.write("To get more information about a mushroom, you need to know at least one of two information: ",
                 "the **family** to which it belongs, and/or its **species**.")
        st.info("There are fungi families that include more than 400 species, with very different physical characteristics."
                "\n To get reliable results, this application will allow you to identify the species only.")
        
if navigation == "Predictions with deep learning":
    with test_images_display:
        
        st.header('Time to identify the Mushrooms !')
        
        #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',unsafe_allow_html = True)

        st.write("To optimize the execution time, only the 15  most present species in the dataset were taken into account.")
        st.info("The 15 species are : Amanita flavoconia, Amanita phalloides, Coprinus comatus, Fomitopsis pinicola, Galerina marginata,"
                          "Ganoderma applanatum, Hypholoma fasciculare, Laetiporus sulphureus, Lycoperdon perlatum, Mycena haematopus,"
                          "Phaeolus schweinitzii, Pleurotus ostreatus, Pluteus cervinus, Schizophyllum commune and Trametes versicolor.")
        
        #@st.cache
        # Function to load the model chosen
        def load_model(model):
            #EfficientNetB0
            if model == models[0]:
                model = tf.keras.models.load_model('MushroomRecoStreamlit.hdf5')
            #VGG16
            elif model == models[1]:
                model = tf.keras.models.load_model('MushroomRecoStreamlitVGG16.hdf5')
            #ResNet50
            else :
                model = tf.keras.models.load_model('MushroomRecoStreamlitResNet5.hdf5')

            return model 
        
        # Function to upload, display and predict the image 
        def import_and_predict(image_user, model):
            size = (224, 224)
            image = ImageOps.fit(image_user, size, Image.ANTIALIAS)
            img = np.asarray(image)
            img_reshape = img[np.newaxis, ...]
            prediction = model.predict(img_reshape)
            
            return prediction
        
        # Function to dispaly images of mushrooms belongig to the same specie
        img_species_path = 'E:\\Jihane 2022\\Projet\\mushroom\\mushroom\\images\\species\\train_test_images'
        def affichage_exemples (classe):
            img_path = img_species_path + "\\" + classe
            image_exemple = random.choice(os.listdir(img_path))
            exemple = Image.open(img_path + '/' + str(image_exemple))
                
            return exemple
        
        # Models available
        models = ["EfficientNetB0", "VGG16", "ResNet50"]
        
        # Model chosen
        st.write("The models available for prediction are:")
        choix_model = st.selectbox("",options = models )
        
        # Model load
        model = load_model(choix_model)
            
        class_species = ["Amanita flavoconia","Amanita phalloides","Coprinus comatus","Fomitopsis pinicola", "Galerina marginata",
                         "Ganoderma applanatum", "Hypholoma fasciculare", "Laetiporus sulphureus", "Lycoperdon perlatum", "Mycena haematopus",
                         "Phaeolus schweinitzii", "Pleurotus ostreatus","Pluteus cervinus", "Schizophyllum commune", "Trametes versicolor"]
        
        # Image upload and display
        st.write("Please upload the image of the mushroom you wish to identify: ")
        file = st.file_uploader("",type = ["jpg" , "png"])
        
        if file is None :
           st.write("Enter here the image !")
        
        else :
            image = Image.open(file)
            st.image(image, use_column_width = True)
            classify = st.button("Predict")
            if classify :
                with st.spinner("Wait for it ..."):
                    st.write("")
                    predictions = import_and_predict(image, model)
                    
                   
                    solution = "It's a " + str(round(np.max(predictions)*100 ,2))+'% ' +class_species[np.argmax(predictions)]
                    
                    
                st.success(solution)
                
                liste_images = []
                st.write("Here are 3 exapmles of other mushrooms of the same specie :")
                for i in range(3):
                    liste_images.append(affichage_exemples(class_species[np.argmax(predictions)]))
            
                with st.spinner("Wait for it ..."):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(liste_images[0], use_column_width = 'always')
                    with col2:
                        st.image(liste_images[1], use_column_width = 'always')
                    with col3:
                        st.image(liste_images[2], use_column_width = 'always')
                    
              
