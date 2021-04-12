import cv2
import numpy as np
import dlib
import face_recognition
detector = dlib.get_frontal_face_detector()
new_path ="/Users/roshanvenkat/PycharmProjects/Hallaxy/venv/cropped_images/"
def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =2):

    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

def save (img,name, bbox, width=180,height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height)) #we need this line to reshape the images
    cv2.imwrite(name+".jpg", imgCrop)

def faces():
    vid = cv2.VideoCapture(0)
    i = 0
    while i < 10:
        ret, frame1 = vid.read()
        i += 1
    vid.release()
    cv2.imwrite("img.jpg", frame1)
    frame =cv2.imread('img.jpg')
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    fit =20
    # detect the face
    for counter,face in enumerate(faces):
        #print(counter)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
        MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
        # save(gray,new_path+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
        save(gray,new_path+str(counter),(x1,y1,x2,y2))
    frame = cv2.resize(frame,(800,800))
    print("done saving")

if __name__ == '__main__':

        PATH = "/Users/roshanvenkat/PycharmProjects/Hallaxy/venv/cropped_images/"
        faces()
        original = cv2.imread("/Users/roshanvenkat/PycharmProjects/Hallaxy/venv/cropped_images/0.jpg")
        duplicate = cv2.imread("/Users/roshanvenkat/PycharmProjects/Hallaxy/venv/cropped_images/1.jpg")
        if original.shape == duplicate.shape:
            print("The images have same size and channels")
            difference = cv2.subtract(original, duplicate)
            print(np.sum(difference))
            w, h, c = difference.shape
            total_pixel_value_count = w * h * c * 255
            percentage_match = (total_pixel_value_count - np.sum(difference)) / total_pixel_value_count * 100
            print(percentage_match)
            if percentage_match > 70:
                print("Image Matched")
            else:
                print("Image Not Matched")