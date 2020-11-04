import face_recognition
from PIL import Image
from pathlib import Path
# this will make it easy to load a folder full of image

# Load the image of the person we want to find similar people for
known_image = face_recognition.load_image_file("person_1.jpg")

# Encode the known image
known_image_encoding = face_recognition.face_encodings(known_image)[0]



# Variables to keep track of the most similar face match we've found
# best face distance keep track of lowest face distance we have seen so far
# intialize it to maximun face distance
best_face_distance = 1.0

# this is where we store actual image of the face thats most similar to what we have processed so far
# initially give it a none value
best_face_image = None

# todo : use pathlib to loop over all the .png file in people folder
for image_path in Path("people").glob("*.png"):
    # Load an image to check
    unknown_image = face_recognition.load_image_file(image_path)

    # Get the location of faces and face encodings for the current image
    face_encodings = face_recognition.face_encodings(unknown_image)

    # todo : write a function to check the difference ebetween known and unknown faces
    face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]


    if face_distance < best_face_distance:
        # Save the new best fac e distance
        best_face_distance = face_distance
        # Extract a copy of the actual face image itself so we can display it
        best_face_image = unknown_image


# Display the face image that we found to be the best match!
pil_image = Image.fromarray(best_face_image)
pil_image.show()
