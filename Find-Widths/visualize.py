import cv2
from matplotlib import pyplot as plt
img_path = "markings.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

plt.imshow(img)
plt.show()