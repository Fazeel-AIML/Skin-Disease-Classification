from django.shortcuts import render, redirect
from django.http import HttpResponse
import tensorflow as tf
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import io
import base64
from tensorflow.keras.preprocessing import image as keras_image

# Load the model
model = tf.keras.models.load_model('skin_cancer_classifier.h5')  # Replace with the path to your saved model

def home(request):
    classification = request.session.get('classification', 'None')
    image_url = request.session.get('image_url', '')

    # Check if no image has been uploaded
    if not image_url:
        default_image_url = 'static/default_image.png'  # Update the path to your default image
        request.session['classification'] = 'None'
        request.session['image_url'] = default_image_url

    return render(request, 'home.html', {'classification': classification, 'image_url': image_url})

def upload_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['upload']

        # Read the image data
        image_data = uploaded_file.read()

        # Decode the image data
        nparr = np.frombuffer(image_data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize the image for classification
        resized_image = cv2.resize(original_image, (224, 224))
        img_array = keras_image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale pixel values to [0, 1]

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted label
        pred_label = 0 if predictions[0][0] > 0.5 else 1

        # Define the class labels
        class_labels = ['Seborrheic', 'Melanoma']

        # Get the classification
        classification = class_labels[pred_label]

        # Encode the original image to base64 format
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Store the classification and image URL in the session
        request.session['classification'] = classification
        request.session['image_url'] = encoded_image

        return redirect('MyApp:home')

    else:
        return HttpResponse("Invalid request method.")
