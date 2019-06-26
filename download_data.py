#!/usr/bin/env python
# coding: utf-8

# In[19]:


# import the necessary packages
from requests import exceptions
import argparse
import requests
import os
import cv2
import matplotlib as plt
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import time


# In[ ]:


def input_options():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
        help="path to download folder")
    ap.add_argument("-n", "--name", required=True,
        help="company's name: azul or gol")
    ap.add_argument("-b", "--batch",
        help="number of photos to download if wanted less than 252",type=int, default=252)
    ap.add_argument("-i", "--iterations",
        help="number of iterations to download per batch",type=int, default=1)
    ap.add_argument("-s", "--startPage",
        help="page's number to start the download",type=str, default='1')
    args = vars(ap.parse_args())
    return args


# In[29]:


#Initializing parameters
def initilize_parameters(args):
    global url, fileN, fileDownloaded, nPhotos, page, batch, header, company, int_page
    fileN = 0
    fileDownloaded = 0
    REQUEST_HEADER = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    header = REQUEST_HEADER
    company = args['name']

    nPhotos = 0
    batch = args['batch']

    StartPage = args['startPage']
    url_azul = 'https://www.airliners.net/search?keywords=azul&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail&page='+StartPage
    url_gol = 'https://www.airliners.net/search?keywords=gol&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail&page='+StartPage
    if company == 'azul':
        url = url_azul
    if company == 'gol':
        url = url_gol

    page = url[-1]
    int_page = int(page)


# In[28]:


def download_batch(args):
    global url, fileN, fileDownloaded, nPhotos, page, batch, header, company, int_page
    path = args['path']
    fileDownloadedperPage = 0
    nPhotos = nPhotos + batch
    while True:
        response = urlopen(Request(url, headers=header))

        # show the page in html code
        soup = BeautifulSoup(response, 'html.parser')

        #find the image link in the html code and download it
        image_elements = soup.find_all("img")
        for img_attr in image_elements:
            if fileDownloaded == nPhotos:
                print("batch downloaded!")
                break
            link = img_attr.attrs['src']
            link_split = link.split("/")
            if link[0:5] == 'https' and link_split[-1][0] != 'p':
                    url2 = link.split("-")[0]+".jpg"
                    r = requests.get(url2)
                    with open(os.path.join(path,company+str(fileN)+".jpg"), 'wb') as file:
                        file.write(r.content)

                    # crop the image to remove airliners.net logo
                    image = cv2.imread(path+"/"+company+str(fileN)+".jpg")
                    x=0
                    y=0
                    w=1200
                    h= image.shape[0] - 50
                    image2 = image[y:y+h, x:x+w]
                    cv2.imwrite(path+"/"+company+str(fileDownloaded)+".jpg",image2)

                    print("Image"+str(fileDownloaded)+" Downloaded")
                    fileDownloaded += 1
                    fileDownloadedperPage += 1
            fileN += 1

        # Follow to the next page when the limit of images is reached (36)   
        if fileDownloadedperPage == 36:
            if url[-2] == '=':
                page = url[-1]
            else:
                page = url[-2] + url[-1]
            print("Current page: "+str(page))
            int_page = int(page) + 1
            url = url.replace('page='+page, 'page='+str(int_page))
            print("Next page: "+str(int_page))
            fileDownloadedperPage = 0

        if fileDownloaded == nPhotos:
            print("Current page: "+str(int_page))
            return True


# In[11]:


def download_iterations(args):
    iterations = args['iterations']
    for i in range(iterations):
        flag= download_batch(args)
        if flag == True:
            time.sleep(10)


# In[12]:


def main():
    args = input_options()
    initilize_parameters(args)
    download_iterations(args)
    print('everything downloaded')
    


# In[16]:


if __name__ == "__main__":
    main()





