import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from annotated_text import annotated_text
from streamlit_option_menu import option_menu
from htbuilder import div
import streamlit.components.v1 as components
st.set_page_config(layout="wide")
st.title('semantic image segmentation')
if 'image_to_show' not in st.session_state:
    st.session_state.image_to_show = None
piece = 1
def get_article(path):
    with open(path, 'r', encoding="utf-8") as file:
        text = file.read()
    return text


def get_keywords(path):
    clean_labels = []
    with open(path, 'r',encoding="utf-8") as file:
        text = file.read()
        labels = text.split("\n")
        for label in labels:
            if label == "":
                continue
            if label[:-1] ==" ":
                clean_labels.append(label[:-1])
            else:
                clean_labels.append(label)
        
    return clean_labels

def create_html(text,keywords):

    for keyword in keywords:
        text = text.replace(keyword, "<a href='#' id='"+keyword+"' class='button-link' onclick='setCompvalue("+keyword+")'>"+keyword+"</a>")
    return text




directory = './jpg'
contents = os.listdir(directory)
with st.sidebar:
    selected = option_menu(menu_title = None, options = ["Painting" + str(i) for i in range(1,len(contents)+1)], menu_icon="cast", default_index=0)
    piece = str(selected[-1])
###########################################################
text = get_article("./articles/"+str(piece)+".txt")
tags = get_keywords('./keyphrases/'+str(piece)+'.txt')
original = Image.open("./jpg/"+str(piece)+".jpg")
new_text = create_html(text,tags)
###########################################################


component = """
<head>
    <style>
        a:link {
        text-decoration: none;
        color: black;
        text-align: center;
        text-decoration: none;
        }
        a:visited {
        text-decoration: none;
        color: black;
        text-align: center;
        text-decoration: none;
        }
        a:hover {
        text-decoration: none;
        color: white;
        }
        a:active {
        text-decoration: none;
        }
        .button-link {
            padding: 2px 2px;
            font-size: 16px;
            color: white;
            background-color: #98FF98; /* Button color */
            text-decoration: none; /* Remove underline */
            border-radius: 10px; /* Rounded edges */
            transition: background-color 0.3s, transform 0.3s;
        }

        .button-link:hover {
            background-color: #228B22; /* Darker shade on hover */
            transform: scale(1.01); /* Slightly enlarge on hover */
        }
    </style>
    
</head>
<body>
    <div> """ +new_text +"""</div>
    <script>
    import React from "react";
    import {
    withStreamlitConnection
    } from "streamlit-component-lib";
    const MyComponent = (props) => {
        const { args } = props;

        const sendValueToStreamlit = (value) => {
            props.setComponentValue(value);
        };
    function setCompvalue(value) {
        const valop = "hahh";
        Streamlit.setComponentValue(valop);
    });
    export default withStreamlitConnection(MyComponent)
    </script>
</body>
"""



col1, col2 = st.columns([2,3])
with col2:
    st.header("Description")
    # var = components.html(component,height = 500)
    # print(var)
    st.write(component, unsafe_allow_html=True)
    for keyword in tags:
        if st.button(keyword):
            selected_image = keyword
with col1:
    st.header("Painting")
    if 'selected_image' in locals():
        try:
            img = Image.open("./founds/"+selected_image+".jpg")
            st.image(img, caption="Selected Image", use_column_width=True)
        except:
            st.write("This part of the image couldn't be identified.")
            st.image(original)
    else:
        st.write("Please select an part of the image from the left.")
        st.image(original)
        
    
    