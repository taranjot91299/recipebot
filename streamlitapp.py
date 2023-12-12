import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from openai import OpenAI


def defmodel():
    from keras.models import Sequential
    from keras.layers import Conv2D,MaxPooling2D
    from keras.layers import Activation, Dense, Flatten, Dropout,Activation,BatchNormalization
    from keras.optimizers import Adamax
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint
    from keras import backend as K

    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = (3,3),input_shape=(100,100,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(131,activation = 'softmax'))
    model.summary()
    return model
# Load your trained model
model = defmodel()
model.compile(loss='categorical_crossentropy',
              optimizer='Adamax',
              metrics=['accuracy'])
model.load_weights('fruits_360_weights_1.hdf5')
class_labels = np.load('target_labels.npy')


# Set your OpenAI GPT-3 API key
client = OpenAI(api_key="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")

# Function to recognize ingredients from an image
def recognize_ingredients(images):
    recognized_ingredients = []
    for uploaded_file in images:
        # Preprocess the image
        img = image.load_img(uploaded_file, target_size=(100, 100))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        # Make predictions
        predictions = model.predict(img_tensor)
        predicted_class = np.argmax(predictions)

        
        # Map the predicted class to the corresponding ingredient
        

        recognized_ingredients.append(class_labels[predicted_class])

    return recognized_ingredients

# Function to generate recipe recommendations using ChatGPT
def generate_recipe_recommendations(ingredients):
    # [Construct input_text and messages as before]
    input_text = f"Generate recipe recommendations using these ingredients: {', '.join(ingredients)}"

    # Prepare the messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text}
    ]
    # Call the OpenAI API for chat completions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the correct model
        messages=messages
    )

    # Print the response to understand its structure
    print(response)

    # Extract the generated recipe recommendation
    # Update the following line according to the actual structure of 'response'
    generated_recipe = response.choices[0].message.content  # Modify this line as per the response structure

    return generated_recipe



# Streamlit app
st.title("Recipe Chatbot")

# Text input for ingredients
text_input = st.text_input("Enter ingredients (comma-separated):")

# File uploader for ingredient images
uploaded_files = st.file_uploader("Upload images of ingredients...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if text_input or uploaded_files:
    # Combine text and image inputs
    if text_input:
        ingredients_from_text = [ingredient.strip() for ingredient in text_input.split(",")]
    else:
        ingredients_from_text = []

    if uploaded_files:
        ingredients_from_image = recognize_ingredients(uploaded_files)
    else:
        ingredients_from_image = []

    all_ingredients = ingredients_from_text + ingredients_from_image

    # Display recognized ingredients
    st.write("Recognized Ingredients:")
    st.write(", ".join(all_ingredients))

    # Generate recipe recommendations based on recognized ingredients using ChatGPT
    recommended_recipes = generate_recipe_recommendations(all_ingredients)

    # Display recommended recipes
    st.write("Recommended Recipes:")
    st.write(recommended_recipes)
