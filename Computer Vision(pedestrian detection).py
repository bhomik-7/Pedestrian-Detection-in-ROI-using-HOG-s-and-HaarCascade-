#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Video

Video("vid.mp4")


# In[2]:


import json
json__ = open('cam_config.json')
data = json.load(json__)


# In[3]:


list__ = []
for i in data['siteConfig']['cameraConfig']['201']['roi']:
    list__.append(i)
json__.close()

list__


# In[4]:


sp1_x = int((list__[1]['coordinates'][0]['x'])*1000)
sp1_y = int((list__[1]['coordinates'][0]['y'])*1000)
sp2_x = int((list__[0]['coordinates'][0]['x'])*1000)
sp2_y = int((list__[0]['coordinates'][0]['x'])*1000)


# In[5]:


ep1_x = int((list__[1]['coordinates'][3]['x'])*1000)
ep1_y = int((list__[1]['coordinates'][3]['y'])*1000)
ep2_x = int((list__[0]['coordinates'][3]['x'])*1000)
ep2_y = int((list__[0]['coordinates'][3]['x'])*1000)


# In[6]:


sp1 = (sp1_x, sp1_y)
sp2 = (sp2_x, sp2_y)
sp2


# In[7]:


ep1 = (ep1_x, ep1_y)
ep2 = (ep2_x, ep2_y)
ep2


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


get_ipython().system('pip list | grep opencv')


# In[10]:


import cv2 as cv


# In[11]:


cap = cv.VideoCapture('vid.mp4')


# In[12]:


frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))


# In[13]:


frame_width


# In[14]:


cap


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Haar Cascade approach to Pedestrian Detection, didn't worked well so I implemented another model

# In[16]:


#haar_cascade = cv.CascadeClassifier('haarcascade_upperbody.xml')
#while(cap.isOpened()):
#    ret, frame = cap.read()
#    if ret == True:
#        resize__ = cv.resize(frame, (1000, 900))
#        rect_1 = cv.rectangle(resize__, sp, ep, (0,255,255), 1)
#        gray = cv.cvtColor(resize__, cv.COLOR_BGR2GRAY)
#        gray = gray[sp_x:ep_x, sp_y:ep_y]
#        faces_rect = haar_cascade.detectMultiScale(gray , scaleFactor = 2.0 , minNeighbors = 5)
#        colored = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
#        for (x , y , w , h) in faces_rect :
#            cv.rectangle(colored, (x,y) , (x+w , y+h) , (255 , 0 , 0) , thickness = 2)
#        cv.imshow('roi', colored)
#        if cv.waitKey(60) & 0xFF == ord('d'):
#            break
#        cv.imshow('cctv footage', resize__)
#        if cv.waitKey(60) & 0xFF == ord('e'):
#            break
#cap.release()
#cv.destroyAllWindows()


# In[17]:


get_ipython().system('pip install imutils')


# Histograms of Oriented Gradients or HOG model detecting pedestrians on the sidewalk and entry gate.

# In[18]:


frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
out1 = cv.VideoWriter('video.avi',cv.VideoWriter_fourcc('M','J','P','G'), 1, (frame_width,frame_height))
out2 = cv.VideoWriter('video.avi',cv.VideoWriter_fourcc('M','J','P','G'), 1, (frame_width,frame_height))
import imutils
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        crop1 = frame[sp1_x:ep1_x, sp1_y:ep1_y]
        crop2 = frame[sp2_x:ep2_x, sp2_y:ep2_y]
        #resize__ = imutils.resize(frame, width=min(500, frame.shape[1]))
        #resize__ = cv.resize(frame, (900, 700))
        #rect_1 = cv.rectangle(crop, sp1, ep1, (0,255,255), 1)
        (regions1, _)= hog.detectMultiScale(crop1, winStride=(4,4), padding = (4,4), scale = 1.05)
        (regions2, _)= hog.detectMultiScale(crop2, winStride=(4,4), padding = (4,4), scale = 1.05)
        for (x, y, w, h) in regions1:
            cv.rectangle(crop1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for (x, y, w, h) in regions2:
            cv.rectangle(crop2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow("entry", crop1)
        cv.imshow("sidewalk", crop2)
        out1.write(crop1)
        out2.write(crop2)
        frametime1 = 1
        frametime2 = 1
        if cv.waitKey(frametime1) & 0xFF == ord('d'):
            break
        if cv.waitKey(frametime2) & 0xFF == ord('d'):
            break
    #print(box)
#out.release()
cap.release()  
cv.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




