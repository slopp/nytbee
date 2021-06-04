from bs4 import BeautifulSoup as bs
import requests
import pytesseract
import io
import cv2
from PIL import Image
import re   
from skimage import io
import pickle
import numpy as np

# open pre-trained image for o
with open("o.pkl", 'rb') as f:
  o_img = pickle.load(f)

with open("q.pkl", 'rb') as f:
  q_img = pickle.load(f)

# download image
#date = "20200613" # X / O test case
#date = "20210402" # too many test case
#date = "20210409" # too many test case
date = "20210515"  # Q test case


bee_url = "https://nytbee.com/Bee_"+date+".html"
host = "https://nytbee.com/"

headers = {'user-agent': 'Chrome'}
page = requests.get(bee_url, headers=headers)

parsed_page = bs(page.content, 'html.parser')
  
  

img_suffix = parsed_page.find(id='bee-pic').img.attrs['src']
img2 = io.imread(host+img_suffix)
orig = img2.copy()
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# find the contours
contours,_=cv2.findContours(gray, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
                            
ocr_letters = list()
for cnt in contours :
    # for each countour rough out the area size
    area = cv2.contourArea(cnt)
    
    # if the area is large its a polygon not a letter, so remove
    if area > 400: 
      approx = cv2.approxPolyDP(cnt, 
                                0.009 * cv2.arcLength(cnt, True), True)
      # remove by drawing this polygon in white
      # if you want to see the large polygons, try uncommenting and then
      # uncomment the final line to draw the whole image
      #cv2.drawContours(img2, [approx], 0,(0, 0, 255), 5)
      cv2.drawContours(img2, [approx], 0,(255, 0, 255), 5)
    
    # if the area is small, then figure out the letter!
    if area<300 and area>50:
      newimg = orig.copy()
      x, y, w, h = cv2.boundingRect(cnt)
      cropped = gray[y:y + h, x:x + w]
      # uncomment to draw a green box around identified letters in img2
      rect = cv2.rectangle(newimg, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.imwrite('temp1.png', newimg)
      cv2.imwrite('temp2.png', cropped)
      print(area)

      # uncomment to see each cropped image
      
      # mode 10 tells tesseract we are looking for letters
      if np.array_equal(cropped, o_img):
        ocr_letters.append('O')
      elif np.array_equal(cropped, q_img):
        ocr_letters.append('Q')
      else:
        ocr_letters.append(pytesseract.image_to_string(cropped, config="--psm 10"))   

# uncomment to see the final cv2 image
cv2.imwrite("ocr_findings.png", img2)

# finally, clean up the identified letters, use | for I ;) 
letters = [re.findall(r"([A-Za-x|]+)", ocr_letter) for ocr_letter in ocr_letters]
# flatten
letters = [letter[0][0] for matches in letters for letter in matches]

print("done")
