import cv2
from pathlib import Path
import os, numpy as np, pandas as pd
import mediapipe as mp
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from functions import mediapipe_detection,extract_images,extract_keypoint,draw_landmarks,display_angle_table,get_angels
from functions import mp_drawing,mp_drawing_styles,mp_holistic,files,PATHS,holistic



for key in PATHS.keys():
    PATHS[key].mkdir(exist_ok=True)

# load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_PATH'])
detection_model = model_builder.build(model_config=configs['model'],is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATHS['CHECKPOINT_PATH'],'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image,shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image,shapes)
    detections = detection_model.postprocess(prediction_dict,shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
IMAGEFILE_PATH = os.path.join(PATHS['test-images'],'2b11bcd1-881e-11ee-bffd-ec5c68664d70.jpg')

def vision(path):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0

    mp_pose = mp.solutions.pose
    shot_num = 0
    with mp_pose.Pose() as pose:
        while cap.isOpened(): 
            ret, frame = cap.read()

            frame_count += 1
            image_np = np.array(frame)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Indicate frame number and fps at the top-left corner
            cv2.putText(image_np, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_np, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_np, f'Shot number: {shot_num}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            box = detections['detection_boxes'][0]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)

            cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)

            # Crop the player region and extract mediapip keypoints
            player_frame = image_np[ymin:ymax, xmin:xmax]
            results = mediapipe_detection(player_frame,holistic)
            keypoints = extract_keypoint(results)
        
            # draw landmarks on keypoints and display angles on screen
            draw_landmarks(player_frame,results)
            if results.pose_landmarks:
                display_angle_table(image_np_with_detections,results)

            image_np_with_detections[ymin:ymax, xmin:xmax] = player_frame
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (1200, 600)))

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    vision('videos/Maxi5Shot.mov')