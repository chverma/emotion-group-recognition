# -*- encoding: UTF-8 -*-
# Get an image from NAO. Display it and save it using PIL.

import sys
import time
sys.path.append("/opt/pynaoqi")
import base64
# Python Image Library
import Image
import binascii
from naoqi import ALProxy

## http://doc.aldebaran.com/1-14/naoqi/vision/alvideodevice-api.html#resolution
def getNAO_image_PIL(IP, PORT):
  """
  First get an image from Nao, then show it on the screen with PIL.
  """

  camProxy = ALProxy("ALVideoDevice", IP, PORT)
  #resolution = 1    # kQVGA (Image of 320*240px)
  resolution = 2 # kVGA ( Image of 640*480px)

  #colorSpace = 11   # kRGBColorSpace (Color RGB)
  #colorSpace = 13   # kBGRColorSpace
  colorSpace = 8   #khsYColorSpace (Grayscale)
  videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)

  t0 = time.time()

  # Get a camera image.
  # image[6] contains the image data passed as an array of ASCII chars.
  naoImage = camProxy.getImageRemote(videoClient)
  #naoImage  = camProxy.getImageLocal(videoClient)
  t1 = time.time()

  # Time the image transfer.
  print "acquisition delay ", t1 - t0

  camProxy.unsubscribe(videoClient)


  # Now we work with the image returned and save it as a PNG  using ImageDraw
  # package.

  # Get the image size and pixel array.
  imageWidth = naoImage[0]
  imageHeight = naoImage[1]
  array = naoImage[6]

  # Create a PIL Image from our pixel array.
  #im = Image.fromstring("RGB", (imageWidth, imageHeight), array)
  #im = Image.fromstring("L", (imageWidth, imageHeight), array)
  #im.save('prova-'+str(t0)+'.png')
  #return binascii.b2a_base64(array)

  return base64.b64encode(array)

