# **README** #

## **How to install** ##

First we have to install some dependencies:

```
#!bash

sudo apt-get update
sudo apt-get install python-pip

pip install --upgrade pip
sudo apt-get install liblapack-dev libboost-python1.58-dev 
sudo apt-get install opencv cmake
sudo pip install dlib # be careful with free memory, consumes ~2GB compiling final step (99%)
sudo pip install pillow matplotlib
sudo apt-get install python-tk

```