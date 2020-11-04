import face_recognition

image_of_person_1 = face_recognition.load_image_file("person_1.jpg")
image_of_person_2 = face_recognition.load_image_file("person_2.jpg")
image_of_person_3 = face_recognition.load_image_file("person_3.jpg")


# this will grab the image
# person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)
# but as we know that we have only one face in the image we can just grab the first element
person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]
person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]


# todo : put all three faces in a array. in this way we will be able to compare all three image with unkown image at once
# Create a list of all known face encodings
known_face_encodings = [
person_1_face_encoding,
person_2_face_encoding,
person_3_face_encoding,
]


# todo :  we got to load the image we got to check
unknown_image = face_recognition.load_image_file("unknown_7.jpg")

# todo : encode the unknown image
# unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# the above piece of code wont detect images with low resolution
# we can break that into 2 steps
# todo : first find the face location and increase the size if image is too small
# number_of_times_to_upsample will be one by default which will doule the size , when we set to two it becomes 4 times bigger
face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample = 2)
# todo : second pass the face location which we just detected yo face encoding
unknown_face_encodings = face_recognition.face_encodings(unknown_image,known_face_locations= face_location)



# insted of just grabing the first face from the picture , like we did for known face lets loop through the photo

for unknown_face_encoding in unknown_face_encodings:
    #lets compare the known and unknown faces
    result = face_recognition.compare_faces(known_face_encodings , unknown_face_encoding , tolerance = 0.6)
    # this will compare the faces and save true or false value in the result


    if result[0]:
        name = "person-1"
    elif result[1]:
        name = "person-2"
    elif result[2]:
        name = "person-3"

    print("found {} in the photo".format(name))

