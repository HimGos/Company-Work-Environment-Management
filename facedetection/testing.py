import streamlit as st
from PIL import Image
import random
import os

# Get the full path to the current directory
current_directory = os.path.abspath(os.path.dirname(__file__))

# Page 1: User input for 'SUCCESS'
st.title("Welcome to the 'SUCCESS' Challenge")

user_input = st.text_input("Type 'SUCCESS' correctly and press Enter:")

if user_input.strip().lower() == "success":
    st.success("You typed 'SUCCESS' correctly!")
    if st.button("Proceed to the next page (Page 2)", key="proceed_button"):
        st.empty()  # Clear the input field and success message
else:
    st.error("Please type 'SUCCESS' correctly.")

# Page 2: Display a random image
if st.button("Proceed to the next page (Page 2)", key="page2_button"):
    st.title("Random Image Page")

    # List of image file names (you can add your own image files)
    image_files = [
        "faces/himz.jpg",
        "faces/tony.jpg",
    ]

    # Select a random image file
    random_image_file = random.choice(image_files)

    # Display the image
    image = Image.open(random_image_file)
    st.image(image, caption="Random Image", use_column_width=True)

# To run the Streamlit app, use the following command in your terminal:
# streamlit run your_app_filename.py
