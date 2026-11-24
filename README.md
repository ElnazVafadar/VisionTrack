# VisionTrack
### Real-Time Object Detection, Tracking, Segmentation & Motion Analysis

VisionTrack is a computer vision system developed in Python that combines real-time object detection, multi-object tracking, instance segmentation, and motion analysis in a unified video processing pipeline.

The system uses YOLO for object detection and segmentation, ByteTrack for object tracking, and OpenCV for image processing, visualization, and motion analysis.

## Key Features

- Object Detection: Detects and classifies objects in video frames using YOLO.
- Object Tracking: Tracks detected objects across frames using ByteTrack and assigns persistent IDs.
- Object Counting: Records object observations and tracks objects crossing a predefined counting line.
- Instance Segmentation: Generates pixel-level masks to separate detected objects from the background.
- Motion Analysis: Estimates object speed, movement direction, and trajectories.
- Data Analysis: Calculates bounding box dimensions and segmentation mask areas.
- Performance Monitoring: Displays source FPS and processing FPS.
- Data Export: Saves detection results in CSV format and processing statistics in JSON.

## Technologies

Technology | Application

Python     Main development language
YOLO      Object detection and segmentation
ByteTrack      Multi-object tracking
OpenCV    Video processing and visualization
NumPy       Numerical and image operations
CSV / JSON      Data storage and reporting

## System Architecture

1. Video Input
2. YOLO Object Detection & Segmentation
3. ByteTrack Object Tracking
4. Motion and Trajectory Analysis
5. Object Counting and Measurement
6. Visualization and Data Export

## Output

The system generates:

- Annotated video with bounding boxes, object IDs, segmentation masks, and motion trajectories.
- CSV file containing detection and tracking information.
- JSON report containing processing statistics and object counts.

## Accuracy and Performance Evaluation

Initial testing was performed using a traffic video containing approximately 25 visible cars.
The system recorded 37 vehicle detections, including some incorrect classifications as buses and trucks.
Based on this initial test, the system demonstrated approximately 70% detection accuracy as a preliminary qualitative estimate. However, this value has not been validated using a labeled benchmark dataset.
The difference between the number of visible vehicles and recorded detections may result from duplicate detections across frames, tracking inconsistencies, or classification errors.
Further evaluation using Precision, Recall, mAP, IDF1, and counting error is required for a reliable performance assessment.

**Initial counting result:**
- Approximate visible cars: 25
- Recorded vehicle detections: 37
- Counting overestimation: approximately 48%

## Project Status

Developed as a practical computer vision project, with an integrated pipeline for object detection, tracking, segmentation, and motion analysis.

## Future Improvements

- Camera calibration for real-world speed estimation
- SORT and ByteTrack performance comparison
- Advanced object counting and region-based analysis
- Tracking accuracy evaluation using benchmark datasets
- Performance optimization for real-time applications
