import PIL.Image
import PIL.ImageDraw
import face_recognition

#load the file into an array / memory
image = face_recognition.load_image_file("people.jpg")
# todo : this loads the image pixel into an array where each pixel in the image is a element in a array




# find all the faces in an image
#face location is a set of coordinates where the face appears in the image
face_locations = face_recognition.face_locations(image)
# todo : we can run the pre trained hog face detector by calling the face location function in the face detection library
# this function will return list of facesvfound in the image
# if no faces found the list will be empty.
# each faces that is returned will contain 4 points.
# these points are the pixel location of the face in an image, given as top , right , left and bottom coordinates


# print the size of the image and how many faces we found in the image
number_of_faces = len(face_locations)
print(" I found {} number of faces in an image".format(number_of_faces))


# to see where the faces are visually present, we will display the image on the screen and draw a box on top of the image
# we will do that using the pil library.
# todo :  convert the image array into PIL formatted image
# Load the image into a Python Image Library object so that we can draw on top of it and display it
pil_image = PIL.Image.fromarray(image)
#todo : copied the image array into a object



#loop through list of faces and check the location of each one
#in python we can break apart the face_location variable.
for face_location in face_locations:
    top, right, bottom, left = face_location
    print(" Face is located at top -> {}, right-> {}, bottom->{}, left->{} \n".format(top,right,bottom,left))

    # draw a box around the image
    # creating our object
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left,top,right,bottom],outline = "red")


# todo : display image on screen
pil_image.show()
