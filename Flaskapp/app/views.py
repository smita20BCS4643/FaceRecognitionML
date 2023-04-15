import os
import cv2
from app.face_recognition import faceRecognitionPipeline


from flask import render_template, request
import matplotlib.image as matimg

UPLOAD_FOLDER ='static/upload'


def index():
    return render_template('index.html')


def app():
    return render_template('app.html')

def genderapp(): 
    if request.method =='POST':
        f = request.files['image_name']
        filename =f.filename
        #save our image in upload folder
        path =os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)
        #get predictions
        pred_image, prediction = faceRecognitionPipeline(path)
        
        #save into folder predict
        pred_filename ='prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        print(prediction)

        #generate report
        report= []
        for i, obj in enumerate(prediction):
            gray_image = obj['roi'] #grayscale image
            eigen_image = obj['eig_img'].reshape(100,100) #eigen image
            gender_name = obj['prediction_name'] #name
            prob_score = round(obj['score']*100,2) #probability score
            
            # save eigen image in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image)
            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image)

            #save report
            report.append([gray_image_name,
                          eig_image_name,
                          gender_name,
                          prob_score])
    return render_template('gender.html',fileupload = True, report = report) #Post request

    return render_template('gender.html') #get request