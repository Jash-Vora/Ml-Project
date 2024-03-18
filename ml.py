import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf

def load_image(img_folder, img):
    img_path = os.path.join(img_folder, img)
    print("Image Path:", img_path)  # Debugging print statement
    try:
        return cv2.imread(img_path)
    except Exception as e:
        print("An error occurred while loading the image:", e)  # Print the error message
        return None



# Function to generate image embeddings
def get_embedding(img_folder, img_name, model):
    img_path = os.path.join(img_folder, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)

# Function to resize image to 128x128 pixels
def resize_image(img, size=(128, 128)):
    return cv2.resize(img, size)

def get_recommender(df, img_folder, model, top_n=16):
    # Generate image embeddings for dataframe
    df_embs = []
    for img_name in df['id']:
        emb = get_embedding(img_folder, str(img_name) + ".jpg", model)
        df_embs.append(emb)

    # Convert embeddings to DataFrame
    df_embs = pd.DataFrame(df_embs)

    # Calculate Cosine Similarity
    cosine_sim = 1 - pairwise_distances(df_embs, metric='cosine')

    # Generate recommendations
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    idx_rec = [i[0] for i in sim_scores]

    print("Type of df.iloc[idx_rec]:", type(df.iloc[idx_rec])) # Debugging print statement
    return df.iloc[0], df.iloc[idx_rec].values.tolist()  # Returning DataFrame rows as list of lists


# Load dataset
DATASET_PATH = "./archive/"
df = pd.read_csv(os.path.join(DATASET_PATH, "styles.csv"))

# Sort the DataFrame by the 'id' column
df.sort_values(by='id', inplace=True)

# Load image data from folder
image_folder = os.path.join(DATASET_PATH, "images")

# Pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Define model for embedding
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# User Input
user_gender = input("Enter gender (Men/Women/Kids/Unisex): ").capitalize()
user_articleType = input("Enter article type: ")
user_baseColour = input("Enter baseColour: ")
user_season = input("Enter season: ")

# Filter the data based on user input
filters = (df['gender'] == user_gender) & \
          (df['articleType'] == user_articleType) & \
          (df['baseColour'] == user_baseColour) & \
          (df['season'] == user_season) 

filtered_df = df[filters].drop(columns=['Unnamed: 10', 'Unnamed: 11'])

# If no exact match is found, try to find similar items
if filtered_df.empty:
    print("No exact match found. Looking for similar items...")
    filtered_df = df.copy()

try:
    # Generate Recommendations
    original_item, recommended_items = get_recommender(filtered_df, image_folder, model)

    # Display Original Item
    original_item_image = load_image(image_folder, str(original_item['id']) + ".jpg")
    original_item_image = resize_image(original_item_image)
    plt.imshow(cv2.cvtColor(original_item_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Item")
    plt.axis('off')
    plt.show()

    # Display Recommended Items in a 4x4 grid
    num_recommended_items = len(recommended_items)
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for idx, item in enumerate(recommended_items):
        row = idx // cols
        col = idx % cols
        recommended_item_image = load_image(image_folder, str(item[0]) + ".jpg")
        recommended_item_image_resized = resize_image(recommended_item_image)
        axes[row, col].imshow(cv2.cvtColor(recommended_item_image_resized, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f"Recommendation {idx+1}")
        axes[row, col].axis('off')

    # Hide empty subplots if there are fewer than 16 recommendations
    for idx in range(num_recommended_items, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

except KeyError as e:
    print("Error:", e)
except Exception as e:
    print("An error occurred:", e)