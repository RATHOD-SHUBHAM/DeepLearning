import PIL.Image
import PIL.ImageDraw
import face_recognition

image = face_recognition.load_image_file("people.jpg")


# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)
# todo  : list contains one entry of each face found in the image. if no face are found the list will be empty.
# else there will be one set of landmark for each face that was in the image
# each face in the list will be a python dictionary object
# key = name of facial feature like left eye , chin and so on
# value = list of X,Y coordinates of the points that corresponds to the facial features.


number_of_faces = len(face_landmarks_list)
print(" i found alomst {} imagaes in the photo".format(number_of_faces))

pil_image = PIL.Image.fromarray(image)
draw = PIL.ImageDraw.Draw(pil_image)

for face_landmark in face_landmarks_list:
    # for each face there will be several features like (eye , nose , mouth etc )
    for name, list_of_points in face_landmark.items():
        # checking how the raw data looks like
        print("\n")
        print(" the {} in the face has following points: {}".format(name,list_of_points))



        # todo : draw a line representing the facial feature we found \
        # line function requires list of funtion, (line olor adn width are optional )
        draw.line(list_of_points , fill = "red" , width = 2)

pil_image.show()
