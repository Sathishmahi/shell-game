# AI-Powered Shell Game Predictor

Ever wondered if technology could add a twist to the classic shell game? This project aims to bring the excitement of the shell game to the digital realm by combining computer vision, object tracking, and machine learning.

## Project Overview

In this project, we use the following technologies and techniques:

- **YOLOv8**: A fine-tuned object detection model is used to identify and track the cups and the hidden ball in each frame of the video.

- **Custom Object Tracking**: We implement a custom tracking system that assigns and updates tracking IDs for the cups as they shuffle and move across frames.

- **Euclidean Distance**: We calculate the Euclidean distance between stored center points of objects in different frames to determine their tracking IDs, ensuring precise object monitoring.

## How It Works

1. **Initial Cup ID Assignment**: In the first frame, tracking IDs (e.g., 1, 2, 3) are assigned to the cups based on their positions, and their center points are recorded as references.

2. **Dynamic Object Tracking**: As the cups shuffle in the video, the AI calculates the Euclidean distance between the stored center points and the cups' current positions to update their IDs.

3. **Final Prediction Zone**: In the final frame, the AI predicts the location of the hidden ball, and the user makes a prediction by selecting a zone (e.g., 1, 2, 3). The AI's prediction and the user's choice are compared to determine the winner.

## Installation and Usage

To run this project, follow these steps:

1. Clone this repository to your local machine.

2. create , activate conda env and install requirements `bash env.sh`  

3. Install the required dependencies using `pip install -r requirements.txt`.

4. Run the project by executing `bash run.sh`.

5. Follow the on-screen instructions to make predictions and enjoy the game.


## Contributing

Contributions are welcome! Feel free to open issues, suggest improvements, or fork this project for your own experiments.

## Contact

For any questions or feedback, you can reach out to the project owner at [sathishofficial456@gmail.com](mailto:sathishofficial456@gmail.com).

Enjoy the game and have fun outsmarting the AI!

## Project Demo-Video

https://github.com/Sathishmahi/shell-game/assets/88724458/dab372b3-9df0-4b9e-92d3-a3c51a8caa1b


