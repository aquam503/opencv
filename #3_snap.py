import sys
import os
import cv2
from videos import whole_face

SCRIPT_PATH = sys.path[0]
os.chdir(SCRIPT_PATH)


GLASSES_PATH = 'pictures/glasses.png'
CROWN_PATH = 'pictures/crown.png'
SOURCE_PATH = 'pictures/picture_cr.jpg'

face_xml_classifier = os.path.join(os.path.dirname(cv2.__file__),
                                   "data", "haarcascade_frontalface_default.xml")
eyepair_xml_classifier = "xml/haarcascade_mcs_eyepair_big.xml"


fd = whole_face.FaceDetector(face_xml_classifier)
pd = whole_face.PairDetector(eyepair_xml_classifier)

crown_transparent = cv2.imread(CROWN_PATH, cv2.IMREAD_UNCHANGED)
#cv2.imshow('crown', crown_transparent)
glasses_transparent = cv2.imread(GLASSES_PATH, -1)
#cv2.imshow('glasses', glasses_transparent)

p = cv2.imread(SOURCE_PATH)
#cv2.imshow('image', p)
cv2.waitKey(0)
cv2.destroyAllWindows()


def put_crown(crown_image, image, face_rects):
    for (x, y, w, h) in face_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        x1 = x
        x2 = x + w
        y1 = y - h // 2
        y2 = y

        if y1 < 0:
            continue

        roi_crown = image[y1:y2, x1:x2]  
        cv2.imshow('roi_crown', roi_crown)
        width_crown = x2 - x1
        height_crown = y2 - y1
        crown_image = cv2.resize(crown_image,
                                 (width_crown, height_crown))
        mask_crown = crown_image[..., -1] 
        cv2.imshow('mask_crown', mask_crown)
        bgr_crown = crown_image[..., 0:-1] 
        cv2.imshow('bgr_crown', bgr_crown)
        back = cv2.bitwise_and(roi_crown, roi_crown,
                               mask=cv2.bitwise_not(mask_crown))

        cv2.imshow('bc', back)
        front = cv2.bitwise_and(bgr_crown, bgr_crown, mask=mask_crown)
        cv2.imshow('fc', front)
        final = cv2.add(front, back)
        cv2.imshow('final_crown', final)
        roi_crown[...] = final


def put_glasses(glasses_image, ALL_ROI, eyepairs):
    for pair, ROI in zip(eyepairs, ALL_ROI):
        if len(pair) == 0:
            continue
        (ex, ey, ew, eh) = pair[0]
        cv2.rectangle(ROI, (ex, ey),
                      (ex + ew, ey + eh), (255, 0, 0), 2)
        x1 = int(ex - ew / 10)
        x2 = int(x1 + ew + ew / 5)
        y1 = int(ey - eh / 4)
        y2 = int(y1 + 1.5 * eh)
        roi_w, roi_h = ROI.shape[:2]
        if x1 < 0 or x2 > roi_w or y1 < 0 or y2 > roi_h:
            continue
        roi_glasses = ROI[y1:y2, x1:x2] 
        cv2.imshow('roi_glasses', roi_glasses)
        width_glasses = x2 - x1
        height_glasses = y2 - y1
        glasses_image = cv2.resize(glasses_image,
                                   (width_glasses, height_glasses))
        mask_glasses = glasses_image[..., -1]
        cv2.imshow('mask_glasses', mask_glasses)
        bgr_glasses = glasses_image[..., 0:-1]
        cv2.imshow('bgr_glasses', bgr_glasses)
        cv2.rectangle(ROI, (x1, y1), (x2, y2), (0, 255, 255), 2)
        back = cv2.bitwise_and(roi_glasses, roi_glasses,
                               mask=cv2.bitwise_not(mask_glasses))
        cv2.imshow('back', back)
        front = cv2.bitwise_and(bgr_glasses, bgr_glasses, mask=mask_glasses)
        cv2.imshow('front', front)
        final = cv2.add(front, back)
        cv2.imshow('fianl_glasses', final)
        roi_glasses[...] = final
face_rects, ROI = fd.detect_faces(p, return_ROI=True)
pair_rects = pd.detect_pair(ROI)
put_glasses(glasses_transparent, ROI, pair_rects)
cv2.imshow('image', p)
cv2.waitKey(0)
cv2.destroyAllWindows()
put_crown(crown_transparent, p, face_rects)

cv2.imshow('image', p)
cv2.waitKey(0)
