![icon](https://github.com/taranjot91299/recipebot/assets/82886384/18dbc59d-be84-4a74-804a-eff5a65b4e4d)
# Recipe-bot 

## Overview
This Streamlit application recommends recipes based on user-provided ingredients. It utilizes a machine learning model to recognize ingredients from images and integrates with OpenAI's GPT model to generate recipe suggestions.

## Features
- **Ingredient Recognition:** Users can upload images of ingredients, and the app will recognize them using a trained deep learning model.
- **Text Input:** Users can also enter ingredients manually via a text input field.
- **Recipe Generation:** Based on recognized ingredients, the app interacts with OpenAI's GPT model to generate and display recipe recommendations.

## Dataset used :

  ### Fruits 360
  - A dataset of images consists of various fruits and vegetables.

  ### About Dataset

  - Total number of images: 90483.
  
  - Training set size: 67692 images (one fruit or vegetable per image).
  
  - Test set size: 22688 images (one fruit or vegetable per image).
  
  - Multi-fruits set size: 103 images (more than one fruit (or fruit class) per image)
  
  - Number of classes: 131 (fruits and vegetables).
  
  - Image size: 100x100 pixels.

You can have this dataset from : https://www.kaggle.com/moltean/fruits or https://mihaioltean.github.io

## Installation
To run this application locally, follow these steps:

1. Clone the repository: `git clone [repository-url]`
2. Navigate to the project directory: `cd [project-directory]`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Streamlit application: `streamlit run app.py`

## Usage
- **Upload Ingredient Images:** Click the 'Upload' button to select and upload images of your ingredients.
- **Enter Ingredients Manually:** Type the names of ingredients into the text field, separated by commas.
- **Get Recipes:** After adding ingredients, click the 'Generate Recipes' button to view the recommended recipes.

## Technologies Used
- Streamlit for the web application framework.
- TensorFlow/Keras for the ingredient recognition model.
- OpenAI API for generating recipe suggestions.

## Contributing
Contributions to this project are welcome. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b [branch-name]`.
3. Make your changes and commit them: `git commit -m '[commit-message]'`.
4. Push to the original branch: `git push origin [project-name]/[location]`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).


