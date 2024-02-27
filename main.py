import argparse
import json
import logging as log
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO



def load_zones(json_path, zone_str):
    """
        Load zones specified in an external json file
        Parameters:
            json_path: path to the json file with defined zones
            zone_str:  name of the zone in the json file
        Returns:
           zones: a list of arrays with zone points
    """
    # load json file
    with open(json_path) as f:
        zones_dict = json.load(f)
    # return a list of zones defined by points
    return np.array(zones_dict[zone_str]["points"], np.int32)

def draw_text(image, text, point, color=(255, 255, 255)) -> None:
    """
    Draws text

    Parameters:
        image: image to draw on
        text: text to draw
        point:
        color: text color
    """
    _, f_width = image.shape[:2]
    
    text_size, _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=2)

    rect_width = text_size[0] + 20
    rect_height = text_size[1] + 20
    rect_x, rect_y = point

    cv2.rectangle(image, pt1=(rect_x, rect_y), pt2=(rect_x + rect_width, rect_y + rect_height), color=(255, 255, 255), thickness=cv2.FILLED)

    text_x = (rect_x + (rect_width - text_size[0]) // 2) - 10
    text_y = (rect_y + (rect_height + text_size[1]) // 2) - 10
    
    cv2.putText(image, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=2, lineType=cv2.LINE_AA)

def get_iou(person_det, object_det):
    #Obtain the Intersection 
    x_left = max(person_det[0], object_det[0])
    y_top = max(person_det[1], object_det[1])
    x_right = min(person_det[2], object_det[2])
    y_bottom = min(person_det[3], object_det[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    person_area = (person_det[2] - person_det[0]) * (person_det[3] - person_det[1])
    obj_area = (object_det[2] - object_det[0]) * (object_det[3] - object_det[1])
    
    return intersection_area / float(person_area + obj_area - intersection_area)

def intersecting_bboxes(bboxes, person_bbox, action_str, label_map: dict):
    #Identify if person and object bounding boxes are intersecting using IOU
    for box in bboxes:
      if box.cls == 0:
          #If it is a person
          try:
              person_bbox.append([box.xyxy[0], box.id.numpy().astype(int)])
          except:
              pass
      elif box.cls != 0 and len(person_bbox) >= 1:
          #If it is not a person and an interaction took place with a person
          for p_bbox in person_bbox:
              if box.cls != 0:
                  result_iou = get_iou(p_bbox[0], box.xyxy[0])
                  if result_iou > 0:
                     try:
                        person_intersection_str = f"Person #{p_bbox[1][0]} interacted with object #{int(box.id[0])} {label_map[int(box.cls[0])]}"
                     except:
                         person_intersection_str = f"Person {p_bbox[1][0]} interacted with object (ID unable to be assigned) {label_map[int(box.cls[0])]}"
                     #log.info(person_intersection_str)
                     person_action_str = action_str + f" by person {p_bbox[1][0]}"
                     return person_action_str

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("source", help="path or url to video source")

    args = parser.parse_args()

    # Load YOLOv8 Model
    models_dir = Path('./model')
    models_dir.mkdir(exist_ok=True)

    DET_MODEL_NAME = "yolov8n"
    det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt', task='detect')
    label_map = det_model.model.names

    # Load our Yolov8 object detection model
    ov_model_path = Path(f"model/{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
    if not ov_model_path.exists():
        # export model to OpenVINO format
        out_dir = det_model.export(format="openvino", dynamic=False, half=True)

    model = YOLO('model/yolov8m_openvino_model/', task='detect')

    print(f"predictor: {model.predictor}, config: {model.cfg}")

    #Load in our sample video
    VID_PATH = args.source
    #Show the dimensions and additional information from the video
    video_info = sv.VideoInfo.from_video_path(VID_PATH)

    polygon = load_zones("zones.json", "test-example-1")
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)
    #Define empty lists to keep track of labels
    original_labels = []
    final_labels = []
    person_bbox = []
    p_items = []
    purchased_items = set(p_items)
    a_items = []
    added_items = set(a_items)

    cv2.namedWindow("display")
    #Iterate through model predictions and tracking results
    for index, result in enumerate(model.track(source=VID_PATH, show=False, stream=True, verbose=True, persist=True)):
        #Define variables to store interactions that are refreshed per frame
        interactions = []
        person_intersection_str = ""

        #Obtain predictions from yolov8 model
        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id < 55]
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        bboxes = result.boxes
        if bboxes.id is not None:
            detections.tracker_id = bboxes.id.cpu().numpy().astype(int)
            
        labels = [
            f'#{tracker_id} {label_map[class_id]} {confidence:0.2f}'
            for _, _, confidence, class_id, tracker_id
            in detections
        ]

        #Annotate the frame with the zone and bounding boxes.
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, labels=labels)
        frame = zone_annotator.annotate(scene=frame)

        objects = [f'#{tracker_id} {label_map[class_id]}' for _, _, confidence, class_id, tracker_id in detections]

        #If this is the first time we run the application,
        #store the objects' labels as they are at the beginning
        if index == 0:
            original_labels = objects
            original_dets = len(detections_filtered)
        else:
            #To identify if an object has been added or removed
            #we'll use the original labels and identify any changes
            final_labels = objects
            new_dets = len(detections_filtered)
            #Identify if an object has been added or removed using Counters
            removed_objects = Counter(original_labels) - Counter(final_labels)
            added_objects = Counter(final_labels) - Counter(original_labels)

            #Create two variables we can increment for drawing text
            draw_txt_ir = 1
            draw_txt_ia = 1
            #Check for objects being added or removed
            if new_dets - original_dets != 0 and len(removed_objects) >= 1:
                #An object has been removed
                for k,v in removed_objects.items():
                    #For each of the objects, check the IOU between a designated object
                    #and a person.
                    if 'person' not in k:
                        removed_object_str = f"{v} {k} removed from zone"
                        removed_action_str = intersecting_bboxes(bboxes, person_bbox, removed_object_str, label_map)
                        if removed_action_str is not None:
                            #If we have determined an interaction with a person,
                            #log the interaction.
                            log.info(removed_action_str)
                            #Add the purchased items to a "receipt" of sorts
                            if removed_object_str not in purchased_items:
                                #print(f"{v} {k}", a_items)
                                #if f"{v} {k}" in a_items:
                                purchased_items.add(f"{v} {k}")
                                p_items.append(f" - {v} {k}")
                        #Draw the result on the screen        
                        draw_text(frame, text=removed_action_str, point=(50, 50 + draw_txt_ir), color=(0, 0, 255))
                        draw_txt_ir += 80
            
            if len(added_objects) >= 1:
                #An object has been added
                for k,v in added_objects.items():
                    #For each of the objects, check the IOU between a designated object
                    #and a person.
                    if 'person' not in k:
                        added_object_str = f"{v} {k} added to zone"
                        added_action_str = intersecting_bboxes(bboxes, person_bbox, added_object_str, label_map)
                        if added_action_str is not None:
                            #If we have determined an interaction with a person,
                            #log the interaction.
                            log.info(added_action_str)
                            if added_object_str not in added_items:
                                added_items.add(added_object_str)
                                a_items.append(added_object_str)
                        #Draw the result on the screen  
                        draw_text(frame, text=added_action_str, point=(50, 300 + draw_txt_ia), color=(0, 128, 0))
                        draw_txt_ia += 80
        
        draw_text(frame, "Receipt: " + str(purchased_items), point=(50, 800), color=(30, 144, 255))
        cv2.imshow("display", frame)

if __name__ == "__main__":
    main()