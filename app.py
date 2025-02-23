import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from ultralytics import YOLO 
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

with open('model_transferL.pkl','rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
        # Load YOLO model
        
        
    if 'file' not in request.files:
      return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
      return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    yolo_model = YOLO('yolov5s.pt')

    # Perform detection
    results = yolo_model(file_path)
    # results = yolo_model('/content/1111.png')

    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    original_image = cv2.imread(file_path)



    rois = []
    labels = []
    class_names = {
        0: 'cardboard',
        1: 'food_waste',
        2: 'glass',
        3: 'metal',
        4: 'paper',
        5: 'plastic',
    }

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()  # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, xyxy)

            # Step 6: Extract the region of interest (ROI) from the image
            roi = original_image[y1:y2, x1:x2]
            rois.append(roi)

            # Step 7: Classify the ROI using your custom classifier
            roi_resized = cv2.resize(roi, (224, 224))  # Resize for your classifier input size
            roi_resized = np.expand_dims(roi_resized, axis=0)  # Add batch dimension
            roi_resized = roi_resized / 255.0  # Normalize if required by your classifier

            # Classify the object
            classification = model.predict(roi_resized)
            class_label = np.argmax(classification)  # Get the predicted class index

            labels.append(class_label)

            # Step 8: Draw the bounding box and label on the original image
            #class_name = f"Class {class_label}"  # Modify based on your class names
            class_name = class_names.get(class_label, 'Unknown')  # Default to 'Unknown' if the class is not in the dictionary

            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 120), 2)

            # Step 9: Show and save the output image with detections
            cv2.imshow('original_image', original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save the result to a file
            cv2.imwrite('/content/output_image.jpg', original_image)

        return render_template('index.html', prediction_text='Predicted class is $ {}'.format(class_name))

@app.route('/other_model')
def other_model():
    return render_template('othermodel.html')


@app.route('/pyspark')
def analysis():
    return render_template('pyspark.html')


if __name__ == "__main__":
    app.run(debug=True)