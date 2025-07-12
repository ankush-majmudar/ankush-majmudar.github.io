import cv2
import os


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


#min # of feature matches
THRESHOLD = 10  

#load all existing caps and their descriptors
existing_images, filenames = load_images_from_folder("caps_collection/")
existing_descriptors = [compute_orb_features(img)[1] for img in existing_images]

#load all new images
new_images, new_filenames = load_images_from_folder("new_caps/")

#go thru each new image and get the orb features
for idx, new_img in enumerate(new_images):
    _, new_desc = compute_orb_features(new_img)
    found_duplicate = False


    for ex_desc, fname in zip(existing_descriptors, filenames):
        if ex_desc is not None and new_desc is not None:
            #match features
            matches = match_features(new_desc, ex_desc)
            #next iteration if we need more matches
            if len(matches) < THRESHOLD:
                continue
            #avg distance between top 10 matches
            avg_distance = sum(m.distance for m in matches[:10]) / len(matches[:10])
            #if avg distance below cutoff then likely a match
            if avg_distance < 35:
                print(f"[!] {new_filenames[idx]} is likely ALREADY in collection as {fname}")
                #save visual files for self inspection
                out_path = f"matches_found/{new_filenames[idx].split('.')[0]}_vs_{fname.split('.')[0]}.jpg"
                save_side_by_side(new_img, existing_images[filenames.index(fname)], out_path)
                found_duplicate = True
                break

    if not found_duplicate:
        print(f"[✓] {new_filenames[idx]} is UNIQUE")
