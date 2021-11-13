#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os


# # SIFT

# In[16]:


img = cv2.imread('DEMOIMGS/holi.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
cv2.imshow('sift_keypoints',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#draw a circle with size of keypoint and it will even show its orientation

#img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('sift_keypoints.jpg',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#sift = cv2.SIFT_create()
#kp, des = sift.compute(gray,None)


# # Fingerprint match using SIFT 

# In[5]:


test_original = cv2.imread("FPDB/100__M_Left_index_finger.bmp") #read the fingerprint input image
cv2.imshow("Original", cv2.resize(test_original, None, fx=1, fy=1)) #show image in a different window
cv2.waitKey(0) #wait for close on output window
cv2.destroyAllWindows() #after closing,delete the window


# In[11]:


for file in [file for file in os.listdir("FPDB")]: #traverse through the database for matching fingerprint
    fingerprint_database_image = cv2.imread('FPDB/'+file) #reading the traversed file
    #SIFT-->SCALE INVARIANT FEATURE TRANSFORM
    sift = cv2.xfeatures2d.SIFT_create() #before SIFT,we need to create a SIFT object 
    
                                                                        #FINDING KEYPOINTS(Detect) AND DESCRIPTORS(Compute)
    keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None) #in the original image 
                                                                            
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None) #in the traversed image
    
    #Matching and comparing both images using the Keypoints
    #FLANN-Fast Library for Approximate Nearest Neighbors Based Matcher 
    #is used for the comparision of the both,which matches features of both finger prints
    #quick and efficient
    #uses Clustering and Search in MultiDimensional Spaces
    
    #knnMatch to implement Image matching and sort the match results
    #k=2-->best 2 matches for each keypoint
    #matches = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5),dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []
   
    for p, q in matches:
        #CHECKING FOR GOOD MATCHES
        if p.distance < 0.1*q.distance:
            match_points.append(p)
        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            keypoints = len(keypoints_1)            
        else:
            keypoints = len(keypoints_2)
            
        #print(keypoints)
        
        #if we have a proper match,show the output of matched file,%match and the matching lines
        if (len(match_points) / keypoints)>0.95:
            print(round(len(match_points) / keypoints * 100,2),"% match!") #Successful matches from the features tracked
            print("Figerprint Matched from database: " + str(file)) 
            result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, keypoints_2, match_points, None) 
            result = cv2.resize(result, None, fx=5.5, fy=5.5)
            cv2.imshow("result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break;


# In[10]:


os.listdir("FPDB")


# In[7]:


#Brute force matching with ORB Descriptors
# the descriptor of one feature in first set and is matched with all other features in second set 
#using some distance calculation. 
#And the closest one is returned.

import matplotlib.pyplot as plt

img1 = cv2.imread('DEMOIMGS/holi.jpg')          # queryImage
img2 = cv2.imread('DEMOIMGS/holi.jpg') # trainImage

# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

plt.imshow(img3),plt.show()


# In[ ]:




