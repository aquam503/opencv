import cv2
from os import path

p1 = cv2.imread('pictures\picture1.jpg')
p2 = cv2.imread('pictures\picture2.jpg')
p3=cv2.imread('pictures\picture3.jpg')

print(path.dirname(cv2.__file__))

xml_classifier = path.join(path.dirname(cv2.__file__),
                           "data", "haarcascade_frontalface_default.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Computer also can recognize gray pictures that human can recognize    
    #also it's faster ,processing just one layer
    face_calssifier = cv2.CascadeClassifier(xml_classifier)
    rects = face_calssifier.detectMultiScale(image=gray,
                                             scaleFactor=1.15,
                                             minNeighbors=5,
                                             minSize=(30, 30))

    return rects
    
def draw(image, rects, title=None):
    print("=" * 30)
    print("i found {} person(s).".format(len(rects)).title())
    print("=" * 30)

    for x, y, w, h in rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if title:
        cv2.imshow(title, image)
        cv2.waitKey(0)

cv2.imshow("p1", p1)
cv2.waitKey(0)
draw(p1, detect_faces(p1), "p1")

cv2.imshow("p2", p2)
cv2.waitKey(0)
draw(p2, detect_faces(p2), "p2")

cv2.imshow("p3", p3)
cv2.waitKey(0)
draw(p3, detect_faces(p3), "p3")

