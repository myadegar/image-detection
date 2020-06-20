import os

from CameraTracker import SingleCameraTracker

######################## Parameters ##########################
# Input video
input_video_file_cam1 = 'left.mp4'
input_video_path_cam1 = os.path.join('Input_Video', input_video_file_cam1)

# Detection
detection_name = 'cascade_rcnn'
detection_speed = 'normal'  # Acceptable values are "normal", "fast", "faster", "fastest" and "flash"
minimum_detection_probability = 20
custom_objects = {'car': True, 'motorcycle': True, 'truck': True, 'bus': True, 'bicycle': True, 'person': True}
frame_detection_interval = 1
display_percentage_probability = False
display_object_name = False
box_border_thickness = 1

# Detect from images
detect_from_images = True
input_image_path = 'Input_Image'
output_image_resized = False
output_image_width = 800
output_image_height = 600
save_image = True

# Tracking
tracker_name = 'deep_sort'
tracking_max_age = 30
tracking_min_hits = 3
tracking_max_iou_distance = 0.7
tracking_nms_max_overlap = 0.5
distance_metric_matching_threshold = 0.2
distance_metric_buffer_size = 100
# feature_model_file = "Tracking/deep_sort/networks/ckpt.t7"
feature_model_file = "Tracking/deep_sort/networks/model.pth.tar-60"

# Outputs
display = True
save_video = False
split_frame = False
log_info = True
write_tracking_info = False
output_frame_per_second = 20
output_video_resized = True
output_video_width = 800
output_video_height = 600
output_image_dir = 'Output_Image'
output_video_dir = 'Output_Video'
output_image_detection_dir = 'Output_Image_Detection'
output_log_dir = 'Output_Log'
zoom_output = False
zoom_bound_cam1 = [1000, 500, 1500, 1520] # top left: x, y, w, h

##################################################
detection_model_file = 'Detection/Detection_Model/cascade_mask_rcnn_hrnetv2p_w32_20e_20190810-76f61cd0.pth'
config_file = 'Detection/mmdetection_master/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e.py'

##################################################
parameters = {
    'input_video_path_cam1': input_video_path_cam1, 'input_image_path': input_image_path,
    'detect_from_images':detect_from_images, 'output_image_resized':output_image_resized,
    'output_image_width':output_image_width, 'output_image_height':output_image_height,
    'save_image':save_image, 'detection_name': detection_name, 'detection_speed': detection_speed,
    'minimum_detection_probability': minimum_detection_probability, 'custom_objects': custom_objects,
    'frame_detection_interval': frame_detection_interval,
    'display_percentage_probability': display_percentage_probability,
    'display_object_name': display_object_name, 'detection_model_file': detection_model_file,
    'box_border_thickness':box_border_thickness, 'config_file': config_file, 'tracker_name': tracker_name,
    'tracking_max_age': tracking_max_age, 'tracking_min_hits': tracking_min_hits,
    'tracking_max_iou_distance': tracking_max_iou_distance, 'tracking_nms_max_overlap': tracking_nms_max_overlap,
    'distance_metric_matching_threshold': distance_metric_matching_threshold,
    'distance_metric_buffer_size': distance_metric_buffer_size, 'feature_model_file': feature_model_file,
     'display': display, 'save_video': save_video,'split_frame': split_frame, 'log_info': log_info,
    'write_tracking_info': write_tracking_info,'output_frame_per_second': output_frame_per_second,
    'output_video_resized': output_video_resized,'output_video_width': output_video_width,
    'output_video_height': output_video_height,'output_image_dir': output_image_dir,
    'output_video_dir': output_video_dir, 'output_image_detection_dir':output_image_detection_dir,
    'output_log_dir': output_log_dir,'zoom_output':zoom_output, 'zoom_bound_cam1':zoom_bound_cam1
            }

##################################################
if __name__ == "__main__":
    cameraTracker = SingleCameraTracker(**parameters)
    cameraTracker()
