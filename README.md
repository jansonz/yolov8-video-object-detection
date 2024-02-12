# Project Name

This project is an POC implementation of video object detection using OpenCV and YOLOv8. The code uses M1 Apple MacBook GPU. In case you have another GPU or want to use CPU, please update the DEVICE variable in `detect_objects.py`.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install and set up the project, follow these steps:

1. Clone the repository: `$ git clone https://https://github.com/jansonz/yolov8-video-object-detection`
2. Navigate to the project directory: `$ cd yolov8-video-object-detection`
3. Install the required dependencies: `$ pip install -r requirements.txt`

## Usage

To use the project, follow these steps:

1. Prepare a video file for object detection.
2. Change the VIDEO_SOURCE path variable in the `detect_objects.py` file to point to the video file. Many sources are supported i.e webcam, a local file or a live stream.
3. Run the scene detection script: `$ python detect_objects.py`. The YOLO ML model will be downloaded automatically.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.