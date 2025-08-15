# Real-Time Accident & Traffic Analysis using YOLOv8

This project presents a comprehensive system for real-time traffic analysis and incident detection using a custom-trained YOLOv8 model. The system is designed to process video streams, identify various events and vehicle types, and apply advanced logic to assess the severity of traffic incidents.

##  Key Features

-   **Multi-Class Object Detection:** Utilizes a powerful YOLOv8 model trained to detect 9 distinct classes, enabling a detailed analysis of traffic scenes.
-   **Advanced Risk Assessment:** Implements custom logic to analyze detections over time and calculate two critical metrics: **Accident Severity** for participants and **Surrounding Severity** for the general area.
-   **Real-Time Capable:** The core inference logic is built for performance and can be easily adapted to process live video feeds from cameras for real-time monitoring applications.
-   **Custom Data Engineering:** Features an in-notebook data pipeline that merges two distinct datasets and re-maps class labels to create a single, robust training source.

---

##  Dataset

The model was trained on a custom-built dataset created by merging two separate sources from Kaggle. This process, documented in the training notebook, allowed for an expanded and more diverse set of detection classes.

The primary datasets used are:

1.  **Car Accident Severity Dataset**: `https://www.kaggle.com/datasets/ahmedmoorsy/car-sevraccid`
2.  **Video Dataset for Anomaly Detection**: `https://www.kaggle.com/datasets/farahalshehhi/videodata`

---

##  Setup & Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YazanAi-Dev3]
    cd [Accident-Detection-YOLOv8]
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```sh
    pip install -r requirements.txt
    ```
You are now ready to open and run the `Accident_Detection_Inference.ipynb` notebook.

---

##  Project Workflow & Real-Time Capabilities

This repository is structured to demonstrate the full lifecycle of the project:

-   **`Training_and_Data_Merging.ipynb`**: This notebook contains the complete data engineering and training pipeline. It shows how the custom dataset was created from the sources above and how the YOLOv8 model was trained.

-   **`Accident_Detection_Inference.ipynb`**: This is the main demonstration notebook. It loads the pre-trained model from the `/model` directory and runs it on sample data to showcase its analytical capabilities.

### Real-Time Application

The inference logic demonstrated in the notebook (processing a video frame-by-frame and analyzing results) is designed to be highly efficient. To adapt this for a live camera feed, the video file input can be replaced with a direct camera stream source (e.g., using OpenCV's `cv2.VideoCapture(0)`).

---

##  Technologies Used

-   Python
-   PyTorch
-   Ultralytics YOLOv8
-   OpenCV
-   Pandas
-   Jupyter Notebook
-   Kaggle API
