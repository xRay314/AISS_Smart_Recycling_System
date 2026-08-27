'''
import cv2

img = cv2.imread("Test.jpeg")

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))