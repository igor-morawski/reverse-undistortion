import numpy as np
import cv2

def draw_bbox(image, x1, y1, x2, y2, caption, color=(203, 232, 0)):
    b = np.array([x1, y1, x2, y2]).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2),  color=color, thickness=5)
    if caption:
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
    return True

def annotate(image, img_anno, transform_fnc=None, draw_org=False, color=(203, 232, 0)):
    img = image.copy()
    for obj in img_anno:
        x1, y1, x2, y2, label = obj
        if draw_org:
            draw_bbox(img, x1, y1, x2, y2, "", (255, 255, 255))
        if transform_fnc:
            x1, y1 = transform_fnc(x1, y1)
            x2, y2 = transform_fnc(x2, y2)
        draw_bbox(img, x1, y1, x2, y2, label, color)
    return img




