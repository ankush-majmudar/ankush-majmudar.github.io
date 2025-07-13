from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

#min # of feature matches
THRESHOLD = 10  

UPLOAD_FOLDER = "new_caps/"
COLLECTION_FOLDER = "caps_collection/"
MATCH_FOLDER = 'static/matches'
MATCH_DISTANCE_CUTOFF = 35


def save_side_by_side(img1, img2, out_path, label1='New', label2='Existing'):
    #resize both images to same height
    height = 300
    img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))
    #add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img1, label1, (10, 25), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img2, label2, (10, 25), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #concatenate horizontally
    side_by_side = cv2.hconcat([img1, img2])
    #make sure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #save the combined image
    cv2.imwrite(out_path, side_by_side)


#load images
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def compute_orb_features(image):
    #create ORB detector
    orb = cv2.ORB_create()
    #extract features
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(desc1, desc2):
    #brute force matcher: compares every descriptor in desc1 
    #to every descriptor in desc2.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    #sort by how "close" the matched features are
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


#orb script wtapped in flask route for web communication
#GET requests: when you first visit the page, it shows the form to upload files.
#POST requests: when you submit/upload files, the app processes them.
@app.route('/', methods=['GET', 'POST'])
def index():
    #status texts
    messages = []
    #side by side imgs
    match_imgs = []

    #if files uploaded
    if request.method == 'POST':
        files = request.files.getlist('files[]')

        for file in files:
            if file:
                filename = secure_filename(file.filename)
                #temporary directory (new_caps)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                #read new image
                new_img = cv2.imread(save_path)
                #ORB features of the new image
                _, new_desc = compute_orb_features(new_img)

                #load existing caps and descriptors
                existing_images, filenames = load_images_from_folder(COLLECTION_FOLDER)
                existing_descriptors = [compute_orb_features(img)[1] for img in existing_images]

                found_duplicate = False
                #for each existing image and descriptor
                for ex_img, ex_desc, fname in zip(existing_images, existing_descriptors, filenames):
                    if ex_desc is not None and new_desc is not None:
                        #feature matching
                        matches = match_features(new_desc, ex_desc)
                        #if not enough matches go to next existing image
                        if len(matches) < THRESHOLD:
                            continue
                        avg_distance = sum(m.distance for m in matches[:10]) / len(matches[:10])
                        #if feature distances simliar then flag as duplicate
                        if avg_distance < MATCH_DISTANCE_CUTOFF:
                            messages.append(f"Flagged! Likely already in collection!")
                            match_img_path = os.path.join(MATCH_FOLDER, f"{filename}_match.jpg")
                            save_side_by_side(new_img, ex_img, match_img_path)
                            match_imgs.append(match_img_path)
                            found_duplicate = True
                            break

                if not found_duplicate:
                    messages.append(f"UNIQUE and added to collection!")
                    os.rename(save_path, os.path.join(COLLECTION_FOLDER, filename))

    #display status texts and matched images to user
    return render_template("index.html", messages=messages, match_imgs=match_imgs)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)