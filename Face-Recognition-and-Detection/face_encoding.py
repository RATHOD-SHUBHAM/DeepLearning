import PIL.Image
import PIL.ImageDraw
import face_recognition

# todo : we take only one image i  which most of the facial features are seen
# todo : load image into memory
image = face_recognition.load_image_file("person.jpg")

# todo :  loacte the face , find the facial features , align the image and then process the image with Neural N/w model
face_encodings = face_recognition.face_encodings(image)
# result of the function is in an array
# each element in the array represents one face that was found in the image



if len(face_encodings) == 0:
    print(" No face was found ")

else :
    # Grab the first face encoding
    # the first image found will be in index zero
    first_face = face_encodings[0]

print(first_face)


# when you run a differnt picture of the same person it should give a value that is close to these value but not the exact value.
# when you run with differrent  peroson the difference between the values should be more.