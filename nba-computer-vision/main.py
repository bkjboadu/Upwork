import tensorflow as tf
import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
import albumentations as A
import mediapipe as mp
from tensorflow.keras.models import load_model

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfMerger


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
# PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord_labelme.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


PATHS = {
    'workspace':Path('workspace'),
    'images':Path('workspace','images'),
    'train-images':Path('workspace','images','train'),
    'test-images':Path('workspace','images','test'),
    'protoc': Path('workspace','protoc'),
    'scripts': Path('workspace','scripts'),
    'pretrained-model':Path('workspace','pretrained_model'),
    'annotation':Path('workspace','annotation'),
    'models': Path('workspace','models'),
    'CHECKPOINT_PATH': Path('workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': Path('workspace','models',CUSTOM_MODEL_NAME,'export'),
    'TFJS_PATH': Path('workspace','models',CUSTOM_MODEL_NAME,'tfjsexport'),
    'TFLITE_PATH': Path('workspace','models',CUSTOM_MODEL_NAME,'tfliteexport'),
}


files = {
    # 'TF_RECORD_SCRIPTS': Path(PATHS['scripts'],TF_RECORD_SCRIPT_NAME),
    'LABELMAP': Path(PATHS['annotation'],LABEL_MAP_NAME),
    'PIPELINE_PATH':os.path.join(str(PATHS['pretrained-model']),'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8','pipeline.config'),
    'CHECKPOINT_PATH': os.path.join(str(PATHS['pretrained-model']),'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8','checkpoint')
}


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


# category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_angels(results):
        mp_pose = mp.solutions.pose
    # Calculate the wrist
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        wrist = np.array([wrist.x, wrist.y, wrist.z])
        index = np.array([index.x, index.y, index.z])
        elbow = np.array([elbow.x, elbow.y, elbow.z])

        v1 = index - wrist
        v2 = elbow - wrist
        wrist_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # wrist_angle = np.degrees(wrist_angle)
        if index[1] < wrist[1]:
          wrist_angle = -wrist_angle


        # Calculate the right elbow angle
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder = np.array([shoulder.x, shoulder.y, shoulder.z])
        elbow = np.array([elbow.x, elbow.y, elbow.z])
        wrist = np.array([wrist.x, wrist.y, wrist.z])
        v1 = shoulder - elbow
        v2 = wrist - elbow
        elbow_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # elbow_angle = np.degrees(elbow_angle)

        # Calculate the right shoulder angle
        hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        hip = np.array([hip.x, hip.y, hip.z])
        v1 = elbow - shoulder
        v2 = hip - shoulder
        shoulder_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # shoulder_angle = np.degrees(shoulder_angle)
        
        # Calculate the hip angle
        knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        knee = np.array([knee.x, knee.y, knee.z])
        v1 = shoulder - hip
        v2 = knee - hip
        hip_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # hip_angle = np.degrees(hip_angle)

        # Calculate the right knee angle
        ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        ankle = np.array([ankle.x, ankle.y, ankle.z])
        v1 = hip - knee
        v2 = ankle - knee
        knee_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # knee_angle = np.degrees(knee_angle)

        # Calculate the right ankle angle
        heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        heel = np.array([heel.x, heel.y, heel.z])
        v1 = knee - ankle
        v2 = heel - ankle
        ankle_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # ankle_angle = np.degrees(ankle_angle)

        return [wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle]


def display_angle_table(frame,results,width,height):
    
    wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle = get_angels(results)

    cv2.rectangle(frame, (width - 600, height - 250), (width, height), (0, 0, 0), cv2.FILLED)
    cv2.line(frame,(width - 600, height - 250), (width - 600, height ), (255,255,255), 4)
    for i in [250,210,170,130,90,50]:
        cv2.line(frame, (width - 600, height - i), (width, height - i), (255,255,255), 4)
    cv2.putText(frame, f'Right wrist angle: {wrist_angle:.2f} degs', (width - 600, height - 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Right elbow angle: {elbow_angle:.2f} degs', (width - 600, height - 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Right shoulder angle: {shoulder_angle:.2f} degs', (width - 600, height - 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Hip angle: {hip_angle:.2f} degs', (width - 600, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Right knee angle: {knee_angle:.2f} degs', (width - 600, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Right ankle angle: {ankle_angle:.2f} degs', (width - 600, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return [wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle]


def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return results

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(10, 250, 80), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(250, 0, 0), thickness=2, circle_radius=2),)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 250, 80), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(250, 0, 250), thickness=2, circle_radius=2),)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(10, 250, 80), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 121, 255), thickness=2, circle_radius=2),)
    

def extract_keypoint(results):
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

def extract_images(video_path,output_folder,frames_to_extract):
    '''
    extracts images from video
    
    Arguments: 
        video_path: this is the path to the video we are extracting image from
        output_file: this is the directory our extracted image will be saved to
        frame_interval_seconds: this is the seconds interval to save each image
    '''

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_on_save = 0

    Path(output_folder).mkdir(exist_ok=True)
    folder_name = Path(video_path).stem

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # current_time_seconds = frame_count / cap.get(cv2.CAP_PROP_FPS)
        # print(current_time_seconds)

        # Save the frame if it's within the desired interval
        if  frame_count in frames_to_extract:

            final_npy_dir = Path(output_folder,folder_name,f"{frame_on_save}.npy")
            image_np = np.array(frame)
        
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

            # for i in range(len(detections['detection_boxes'])):
            box = detections['detection_boxes'][0]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)

            player_frame = image_np[ymin:ymax, xmin:xmax]
            output_path = Path(output_folder, f"{str(uuid.uuid1())}.jpg")
            print(output_path)
            cv2.imwrite(str(output_path), player_frame)
    cap.release()

def view_fn(path,starting_point):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    frame_to_save = 0
    folder_name = Path(path).stem
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            image_np = np.array(frame)
            
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

            # for i in range(len(detections['detection_boxes'])):
            box = detections['detection_boxes'][0]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)

            player_frame = image_np[ymin:ymax, xmin:xmax]
            
            margin = 50

            text_x = max(min(xmin - margin, player_frame.shape[0] - 150), 0)
            text_y = max(min(ymin - margin, player_frame.shape[1] - 150), 0)

            
            if text_x < player_frame.shape[1] and text_y < player_frame.shape[0]:
                cv2.putText(player_frame, f'{frame_count}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print("Text coordinates exceed frame dimensions.")
            cv2.imshow('frame',player_frame)
            

            if frame_count in np.arange(starting_point,starting_point + 25):
                image,results = mediapipe_detection(player_frame,holistic)
                extracted_results = extract_keypoint(results)
                final_npy_dir = Path('shot_detection_images','0',folder_name,f"{frame_to_save}.npy")
                Path('shot_detection_images','0',folder_name).mkdir(exist_ok=True)
                np.save(final_npy_dir,extracted_results)
                draw_landmarks(image,results)
                mp_drawing.draw_landmarks(player_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(10, 250, 80), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 0, 0), thickness=2, circle_radius=2),)
                frame_to_save += 1
            elif frame_count > starting_point + 25:
                break

        cap.release()
        cv2.destroyAllWindows()

def create_dataframe(keypoints,name_to_save):
    df = pd.DataFrame(keypoints,columns=['wrist_angle','elbow_angle','shoulder_angle','hip_angle','knee_angle','ankle_angle'])
    df.index = range(1, len(df) + 1)
    df.to_excel(f"{name_to_save}.xlsx")
    
def plot_angle_graph_and_save(frames,angles,save_as):

    # Given data
    labels = ["wrist_angle","elbow_angle","shoulder_angle","hip_angle","knee_angle","ankle_angle"]

    df = pd.DataFrame(angles,index=frames,columns=labels)

    # Plotting the data
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('frame')
    plt.ylabel('angles')
    plt.title('joint flexion angles')
    plt.legend()

    # Show the plot
    plt.savefig(save_as)
    plt.close()

def resize_image(img):
    target_width = 100

    height, width, _ = img.shape
    aspect_ratio = height / width

    target_height = int(target_width * aspect_ratio)

    resized_img = cv2.resize(img, (target_width, target_height))

    return resized_img




def create_pdf(pdf_filename, image_paths, folder_name):
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    # Add a header with the folder name
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f'Shot: #{folder_name}')


    # Draw the first image on the left side
    c.drawImage(image_paths[0], 100, 500, width=80, height=100)

    # Draw the second image on the right side
    c.drawImage(image_paths[1], 200, 500, width=200, height=200)

    c.showPage()
    c.save()

def create_combined_pdf(main_folder, pdf_filename):
    merger = PdfMerger()

    # Iterate through subfolders
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # Check if the item in the main folder is a subfolder
        if os.path.isdir(subfolder_path):

            # List image files in the subfolder
            image_files = [file for file in os.listdir(subfolder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Check if there are exactly 2 image files
            if len(image_files) == 2:
                image_paths = [os.path.join(subfolder_path, file) for file in image_files]
                
                # sheet_filename = f'{subfolder}.pdf'
                sheet_filename = os.path.join(subfolder_path,f'{subfolder}.pdf')

                # Create a PDF for each subfolder
                create_pdf(sheet_filename, image_paths, subfolder)
                print(f'PDF created for {subfolder}')

                # Append the sheet to the main PDF
                merger.append(sheet_filename)
            else:
                print(f'Subfolder {subfolder} does not contain 2 images.')

    # Save the final PDF
    merger.write(pdf_filename)
    merger.close()

    # Remove individual subfolder PDFs
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            pdf_path = os.path.join(main_folder,subfolder_path, f'{subfolder}.pdf')
            print(pdf_path)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    print('PDF creation complete.')


def get_shot_pose_image(frame_num,player_angles,player_frame):
    assert len(frame_num) == len(player_angles) == len(player_frame)
    cols = ['wrist_angle','elbow_angle','shoulder_angle','hip_angle','knee_angle','ankle_angle']
    df = pd.DataFrame(player_angles,index=np.arange(len(frame_num)).tolist(),columns=cols)
    elbow_max_index = df['elbow_angle'].idxmax()
    return elbow_max_index


def do_analysis(path):
   cap = cv2.VideoCapture(path)
   fps = cap.get(cv2.CAP_PROP_FPS)
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   frame_count = 0
   actions = {0:'noshot',1:'shot'}

   BUFFER_SIZE = 20 if fps < 70 else 30  # Adjust as needed
   WAITING_BUFFER_SIZE = BUFFER_SIZE + 10
   THRESHOLD = 10 if fps < 70 else 25 # Adjust as needed
   shot_frame_count_max = 12 if fps < 70 else 10
   shot_buffer = []
   waiting_buffer = []

   keypoints_per_frame = []
   player_image_frame = []
   at_shot_frame_num = []
   at_shot_frame_angles = []

   model = load_model('shot_detection_lstm.h5')
   shot_frame_count= 0
   sequences = []
   mp_pose = mp.solutions.pose
   shot_num = 0
   with mp_pose.Pose() as pose:
      while cap.isOpened(): 
            ret, frame = cap.read()

            if not ret:
               create_dataframe(keypoints_per_frame,f"{Path(path).stem}")
               break

            frame_count += 1
            at_shot_frame_num.append(frame_count)
            image_np = np.array(frame)
            

            # Indicate frame number and fps at the top-left corner
            cv2.putText(image_np, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_np, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_np, f'Shot number: {shot_num}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_np, f'Shot frame count: {shot_frame_count}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            
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
            player_frame_to_save = player_frame.copy()
            player_image_frame.append(player_frame_to_save)
            results = mediapipe_detection(player_frame,holistic)
            keypoints = extract_keypoint(results)

            draw_landmarks(player_frame,results)
            if results.pose_landmarks:
               wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle = display_angle_table(image_np_with_detections,results,width,height)
               item_list = [wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle]
               at_shot_frame_angles.append(item_list)
            else:
               item_list = [0,0,0,0,0,0]
               at_shot_frame_angles.append(item_list)

               
            sequences.append(keypoints)
            sequence = sequences[-5:]

            if len(sequence) == 5:
               y_predict = model.predict(np.expand_dims(sequence,axis=0))
               y_predict = np.argmax(y_predict)
               shot_label = actions[y_predict]
               shot_buffer.append(shot_label)
               waiting_buffer.append(shot_label)

               if len(shot_buffer) > BUFFER_SIZE:
                  shot_buffer.pop(0)

               if len(waiting_buffer) > WAITING_BUFFER_SIZE:
                  waiting_buffer.pop(0)

               if shot_buffer.count('shot') >= THRESHOLD:
                  shot_frame_count += 1
               # else:
               #    shot_frame_count = 0
               

               if shot_frame_count >= shot_frame_count_max:
                  # Check the last BUFFER_SIZE frames to see if they all predict "no_shot"
                  if waiting_buffer[-WAITING_BUFFER_SIZE:].count('noshot') == WAITING_BUFFER_SIZE:
                     shot_num += 1
                  # if shot_buffer[-BUFFER_SIZE:].count('noshot') == BUFFER_SIZE:
                  #    shot_num += 1

                     print(len(at_shot_frame_num))
                     print(len(at_shot_frame_angles))
                     print(len(player_image_frame))
                     max_elbow_index = get_shot_pose_image(at_shot_frame_num,at_shot_frame_angles,player_image_frame)
                     
                     frames_to_plot = at_shot_frame_num[max_elbow_index-50:max_elbow_index+50]
                     angles_to_plot = at_shot_frame_angles[max_elbow_index-50:max_elbow_index+50]

                     Path('report').mkdir(exist_ok=True)
                     at_shot_dir = Path('report') / Path(path).stem / f"{shot_num}"
                     at_shot_dir.mkdir(parents=True,exist_ok=True)
                     img_path = at_shot_dir / f"{shot_num}.jpg"
                     graph_path = at_shot_dir / f"graph_{shot_num}.png"
                     shot_frame_count = 0

                     plot_angle_graph_and_save(frames_to_plot,angles_to_plot,str(graph_path))
                     cv2.imwrite(str(img_path),player_image_frame[max_elbow_index])
                     at_shot_frame_num = []
                     at_shot_frame_angles = []
                     player_image_frame = []
                     shot_frame_count = 0
               else:
                  if shot_buffer[-BUFFER_SIZE:].count('noshot') == BUFFER_SIZE:
                     shot_frame_count = 0

            image_np_with_detections[ymin:ymax, xmin:xmax] = player_frame
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (1200, 600)))

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
      cap.release()
      cv2.destroyAllWindows()
               
      pdf_init_path = os.path.join('report',f"{str(Path(path).stem)}")
      create_combined_pdf(pdf_init_path, os.path.join(pdf_init_path,f"{str(Path(path).stem)}_shot_report.pdf"))

      

do_analysis('videos/CoreyHawkins5Shots.mov')

