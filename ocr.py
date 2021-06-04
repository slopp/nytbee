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

def parse_bee_image(parsed_page, host = "https://nytbee.com/"):
  # open files for special cases
  with open("o.pkl", 'rb') as f:
    o_img = pickle.load(f)

  with open("q.pkl", 'rb') as f:
    q_img = pickle.load(f)

  # download image
  img_suffix = parsed_page.find(id='bee-pic').img.attrs['src']
  img2 = io.imread(host+img_suffix)
  gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  # find the contours
  contours,_=cv2.findContours(gray, cv2.RETR_TREE,
                              cv2.CHAIN_APPROX_SIMPLE)
                              
  ocr_letters = list()
  for cnt in contours :
      # for each countour rough out the area size
      area = cv2.contourArea(cnt)
      
      # if the area is large its a polygon not a letter, so remove
      #if area > 400: 
      #  approx = cv2.approxPolyDP(cnt, 
      #                            0.009 * cv2.arcLength(cnt, True), True)
        # remove by drawing this polygon in white
        # if you want to see the large polygons, try uncommenting and then
        # uncomment the final line to draw the whole image
        #cv2.drawContours(img2, [approx], 0,(0, 0, 255), 5)
      #  cv2.drawContours(img2, [approx], 0,(255, 255, 255), 5)
      
      # if the area is small, then figure out the letter!
      if area <300 and area>50:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = gray[y:y + h, x:x + w]
        # uncomment to draw a green box around identified letters in img2
        #rect = cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
        # uncomment to see each cropped image
        # cv2.imwrite('temp'+str(len(l))+'.png', cropped)
        # mode 10 tells tesseract we are looking for letters
        if np.array_equal(cropped, o_img):
          ocr_letters.append('O')
        elif np.array_equal(cropped, q_img):
          ocr_letters.append('Q')
        else:
          ocr_letters.append(pytesseract.image_to_string(cropped, config="--psm 10"))
  
  # uncomment to see the final cv2 image
  # cv2.imwrite("ocr_findings.png", img2)
  
  # finally, clean up the identified letters, use | for I ;) 
  letters = [re.findall(r"([A-Z|]+)", ocr_letter) for ocr_letter in ocr_letters]
  # flatten
  letters = [letter[0][0] for matches in letters for letter in matches]
  return letters

def parse_colon(value):
  val = value.split(":")[1]
  return int(val)


def parse_page(bee_url):
  headers = {'user-agent': 'Chrome'}
  page = requests.get(bee_url, headers=headers)
  if page.status_code != 200:
    print(bee_url+" failed with "+page.status_code)
    return None
  parsed_page = bs(page.content, 'html.parser')
  key_vals = parsed_page.find_all('h3')[0:4]
  vals = [parse_colon(v.string) for v in key_vals]
  letters = parse_bee_image(parsed_page)
  if len(letters) != 7:
    print("Error with OCR on: "+bee_url+" parsed: "+str(letters))
  parsed = {
    'date': parsed_page.find_all('h2')[1].string,
    'num_pangram': vals[0],
    'max_score': vals[1],
    'max_words': vals[2],
    'min_genius': vals[3],
    'letters': letters,
    'req_letter': letters[3]
  }
  return parsed


import pandas as pd
from datetime import date, timedelta
ndays = 500
days = [ (date.today() - timedelta(days=i)) for i in range(ndays)]
urls = [ 'https://nytbee.com/Bee_'+d.__format__('%Y%m%d')+'.html' for d in days]  
results = pd.DataFrame(columns = ['date', 'num_pangram', 'max_score', 'max_words', 'min_genius','letters'])
c = 0
for u in urls:
  parsed = parse_page(u)
  parsed_df = pd.DataFrame.from_dict(parsed)
  results = results.append(parsed_df, ignore_index=True)
  c = c+1
  if c%5==0:
    print("On "+str(c)+" of "+str(ndays))
  
  

results.to_csv("nytbee.csv")


