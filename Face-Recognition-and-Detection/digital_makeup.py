import PIL.ImageDraw
import PIL.Image
import face_recognition

# todo : load the image into array
image = face_recognition.load_image_file("unknown_8.jpg")

# todo : facial features
facial_landmark_list = face_recognition.face_landmarks(image)


# todo :  convert the image array into PIL formatted image
pil_image = PIL.Image.fromarray(image)


# todo : access the drawing tools
# rgba will tell pil we want to treat out image as 4 channel
# rgb is red blue and green and A is for transperacy
draw = PIL.ImageDraw.Draw(pil_image , 'RGBA')

for face_landmark in facial_landmark_list:
    # for each face there will be several features like (eye , nose , mouth etc )
    # todo : draw the eye brow line
    #since the list is a key value pair we can pass the key value
    # eg : left eye brow , chin
    # fill takes in the rgba value 128,0,128 is purple
    # width gives the width of the line
    draw.line(face_landmark["left_eyebrow"], fill = (128 , 0 , 128 , 100), width = 3)
    draw.line(face_landmark["right_eyebrow"], fill = (128 , 0 , 128 , 100), width = 3)
    # todo : for lips we use the polygon function
    # since these are polygon we dont need line width
    draw.polygon(face_landmark["top_lip"], fill = (128 , 0 , 128 , 100))
    draw.polygon(face_landmark["bottom_lip"], fill = (128 , 0 , 128 , 100))


pil_image.show()