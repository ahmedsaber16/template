#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:
import h5py

# Define a custom loss function for Vgg19 UNet model

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_pos = K.cast(K.flatten(y_pred), dtype='float32')
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


# In[2]:


def display_images_with_masks(df):
    for i, row in df.iterrows():
        # Read the image
        img = io.imread(row['image_path'])

        # Plot the original image
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original MRI')

        # Plot the predicted mask
        plt.subplot(1, 3, 2)
        if row['has_mask'] == 1:
            # If there is a mask, display it
            if not isinstance(row['predicted_mask'], str) and row['predicted_mask'] is not None:
                mask = np.array(row['predicted_mask']).squeeze().round()
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Resize mask to match image dimensions
                plt.imshow(mask_resized, cmap='gray')
                plt.title('Predicted Mask')
            else:
                plt.imshow(np.zeros_like(img), cmap='gray')
                plt.title('No Mask Predicted')
        else:
            plt.imshow(np.zeros_like(img), cmap='gray')
            plt.title('No Tumor')

        # Overlay the original MRI with the predicted mask
        plt.subplot(1, 3, 3)
        if row['has_mask'] == 1:
            img_with_mask = np.copy(img)
            if not isinstance(row['predicted_mask'], str) and row['predicted_mask'] is not None:
                mask = np.array(row['predicted_mask']).squeeze().round()
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Resize mask to match image dimensions
                img_with_mask[mask_resized == 1] = [255, 0, 0]  # Overlay red color where mask is predicted
            plt.imshow(img_with_mask)
            plt.title('MRI with Predicted Mask')
        else:
            plt.imshow(img)
            plt.title('No Tumor')


# In[ ]:


from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Function to predict tumor type
def predict_tumor_type(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    preds = model_cls.predict(img)
    return labels[np.argmax(preds)]

# Load segmentation model
model_seg = load_model("seg_model.h5", custom_objects={"focal_tversky": focal_tversky, "tversky": tversky, "tversky_loss": tversky_loss})
model_cls = load_model('modelFineT.h5')
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Function to predict tumor type
def predict_tumor_type(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    preds = model_cls.predict(img)
    return labels[np.argmax(preds)]

# Function to perform segmentation
def segmentation(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))
    img_standardized = (img_resized - img_resized.mean()) / img_resized.std()
    X = np.expand_dims(img_standardized, axis=0)
    predict = model_seg.predict(X)

    if predict.round().astype(int).sum() == 0:
        has_mask = False
        mri_with_mask = None
    else:
        has_mask = True
        img_with_mask = np.copy(img_resized)
        mask_resized = cv2.resize(predict.squeeze().round().astype(np.uint8), (img_resized.shape[1], img_resized.shape[0]))
        img_with_mask[mask_resized == 1] = [255, 0, 0]  # Overlay red color where mask is predicted
        _, img_encoded = cv2.imencode('.jpg', img_with_mask)
        mri_with_mask = base64.b64encode(img_encoded).decode()

    return has_mask, mri_with_mask

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        tumor_type = predict_tumor_type(filepath)
        return jsonify({'result': tumor_type})

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Predict tumor type first
        tumor_type = predict_tumor_type(filepath)
        if tumor_type == 'no_tumor':
            return jsonify({'has_mask': False, 'message': 'No tumor found'})

        # Perform segmentation if tumor is present
        has_mask, mri_with_mask = segmentation(filepath)
        
        return jsonify({'has_mask': has_mask, 'mri_with_mask': mri_with_mask})

if __name__ == '__main__':
    app.run()


# In[ ]:





# In[ ]:
