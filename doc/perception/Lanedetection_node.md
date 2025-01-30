# Overview of the Lanedetection Node

This Document gives a briev overview over the Lanedetection Node.
The Lanedetection Node is responsible for detecting the Lane markings and driveable Area in a given Image and publish the results.

---

## Lanedetection Node System Diagram

![Lanedetection Node System Diagram](../assets/perception/Overview_lanedetection_node.jpg)
(link to lucid chart: [lucid chart](https://lucid.app/lucidchart/34e9aa95-5fb3-4d83-b53f-6d6a3f4748c2/edit?viewport_loc=5190%2C-3952%2C1690%2C703%2C0_0&invitationId=inv_83e27eed-e730-4607-836b-0e863cd2b511))

## Inputs

1. **Center/Image**  
   - **Type**: Image Message (ImageMsg)  
   - **Description**: Provides the center camera image.  

---

## **Processing Steps**

 1. **Image Handler**

    - **Preprocessing**  
        - *Description*: Preprocesses the camera image for the model.  
    - **YOLOP-Model**  
        - *Function*: Utilizes the YOLOP model to generate **Driveable Area** and **Lane Mask**.  
        - **Outputs**:  
            - Driveable Area  
            - Lane Mask
    - **Postprocessing**
        - *Function*: Refines the Lane Mask for better visualization and integration with Lidar data.
        - *Output*: Lane Mask

---

## Outputs

1. **driveable_area**  
   - **Type**: Image Message (ImageMsg)  
   - **Description**: Provides a Mask with 0 = not driveable and 1 = driveable.  

2. **Lane_detection_mask**  
   - **Type**: Image Message (ImageMsg)
   - **Description**: Contains the mask for all detected lane markings.
