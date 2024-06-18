import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load the pre-trained model
model = load_model('fashion_mnist_model.h5')
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Helper functions for displaying images and predictions
def plot_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Streamlit app
st.title('Fashion MNIST Classifier')
st.write("""
This is a web app to classify images of clothing from the Fashion MNIST dataset.
You can upload your own grayscale images to see how well the model performs.
""")

# Upload image
uploaded_file = st.file_uploader("Choose a grayscale image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image_np = np.array(image) / 255.0

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Predict
    img_array = np.expand_dims(image_np, axis=0)
    predictions = probability_model.predict(img_array)
    predicted_label = np.argmax(predictions)

    st.write(f"Prediction: {class_names[predicted_label]}")
    st.write(f"Confidence: {100 * np.max(predictions):.2f}%")

    # Plot the image and prediction
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(predictions[0], predicted_label, image_np)
    plt.subplot(1, 2, 2)
    plot_value_array(predictions[0], predicted_label)
    st.pyplot(plt)

st.write("### Recommended Dataset Source")
if st.button("Go to MNIST Fashion Dataset Source"):
    st.write("Redirecting to [MNIST Fashion Dataset](https://github.com/zalandoresearch/fashion-mnist)...")
    st.markdown("[MNIST Fashion Dataset](https://github.com/zalandoresearch/fashion-mnist)")

# Display some example images
st.write("## Example Images")
st.write("Here are some example images from the Fashion MNIST dataset:")

selected_example_index = st.selectbox(
    "Select an example image to classify:",
    list(range(100)),
    format_func=lambda x: class_names[train_labels[x]]
)

if st.button("Classify selected example image"):
    example_image = train_images[selected_example_index]
    example_image_np = np.array(example_image)
    example_image_expanded = np.expand_dims(example_image_np, axis=0)
    example_predictions = probability_model.predict(example_image_expanded)
    example_predicted_label = np.argmax(example_predictions)

    st.image(example_image, caption=f'Example Image: {class_names[train_labels[selected_example_index]]}', use_column_width=True)
    st.write(f"Prediction: {class_names[example_predicted_label]}")
    st.write(f"Confidence: {100 * np.max(example_predictions):.2f}%")

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(example_predictions[0], train_labels[selected_example_index], example_image_np)
    plt.subplot(1, 2, 2)
    plot_value_array(example_predictions[0], train_labels[selected_example_index])
    st.pyplot(plt)

# Footer with hyperlinks
st.write("Created by [Rauf](https://personal-web-page-lemon.vercel.app/index.html)")

# Add a button for the electricity consumption prediction
st.markdown(
    """
    <style>
    .fixed-footer-button {
        color: #FF6347;
        position: fixed;
        left: 10px;
        bottom: 10px;
    }
    .fixed-footer-button a {
        text-decoration: none;
        color: #FF6347;
    }
    .fixed-footer-button a button {
        background-color: #d41738;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .fixed-footer-button a button:hover {
        background-color: #ff06bd;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="fixed-footer-button">
        <a href="https://raufjatoi-elecustom.streamlit.app/" target="_blank">
            <button>
                Try Electricity Consumption Prediction
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
