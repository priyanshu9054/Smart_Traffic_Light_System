# Smart_Traffic_Light_System

Introduction
This project implements a Smart Traffic Light System using YOLOv8 (You Only Look Once version 8). The system detects and analyzes traffic conditions using real-time object detection, making traffic light control adaptive and responsive to the current traffic situation.

Requirements
ultralytics/YOLOv8
Python 3.x
OpenCV
Traffic camera or webcam

Download YOLOv8 Pre-trained Weights:

Download the YOLOv8 pre-trained weights from the official YOLOv8 GitHub releases and place them in the weights directory.
Run Traffic Light System:

Here sort.py is used as a tracker to count every car only once. 

SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai


Usage
Camera Input:

Connect a traffic camera or webcam to the system.
Real-time Object Detection:

YOLOv8 will detect vehicles and analyze traffic conditions.
Traffic Light Control:

The smart traffic light system adapts the signal timings based on detected traffic, optimizing traffic flow.
Configuration
Adjust parameters in the traffic_light_system.py script to fine-tune the system according to specific traffic scenarios.
File Structure
yolov8/: YOLOv8 repository.
traffic_light_system.py: Python script for the smart traffic light system.
Dependencies
YOLOv8: Real-time object detection model.
OpenCV: Computer vision library for image processing.
