I am delighted to present the initial phase of a project I recently undertook, centered around the application of computer vision techniques to gauge joint flexion in basketball players during shooting sequences. The primary objective is to derive essential metrics from player movements, shedding light on factors influencing accurate shot execution. This installment comprises five distinct components:

Player Detection: Leveraging the TensorFlow Object Detection API, a robust model is developed to accurately identify the basketball player undertaking the shot.


Shot Detection: This phase involves pinpointing the initiation of a shot, a critical aspect for subsequent in-depth analysis.

Ball-Release Detection: A dedicated model is implemented to precisely determine the point at which the player releases the ball, focusing on the last point of contact between the palm and the ball during the shooting motion.

Mediapipe Holistic: Utilizing the Mediapipe Holistic library, key points of the player are extracted, facilitating the calculation of joint angles. These angles are pivotal in creating graphical representations for a comprehensive analysis.

Automated Reporting: A dynamic PDF report is generated automatically after processing the video footage. Each page corresponds to a shot, featuring an image of the player during the shot alongside a graph illustrating the joint angles for detailed insights.

