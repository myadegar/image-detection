
import numpy as np

from Detection.mmdetection_master.mmdet.apis import init_detector
import torch
from Detection.mmdetection_master.mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


class ObjectDetection:
    """
                    This is the object detection class for images in the ImageAI library. It provides support for RetinaNet
                     , YOLOv3 and TinyYOLOv3 object detection networks . After instantiating this class, you can set it's properties and
                     make object detections using it's pre-defined functions.

                     The following functions are required to be called before object detection can be made
                     * setModelPath()
                     * At least of of the following and it must correspond to the model set in the setModelPath()
                      [setModelTypeAsRetinaNet(), setModelTypeAsYOLOv3(), setModelTypeAsTinyYOLOv3()]
                     * loadModel() [This must be called once only before performing object detection]

                     Once the above functions have been called, you can call the detectObjectsFromImage() function of
                     the object detection instance object at anytime to obtain observable objects in any image.
    """

    def __init__(self, gpu_num=0):
        torch.cuda.set_device(gpu_num)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []

        self.__input_image_min = 1333
        self.__input_image_max = 800

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair dryer',
                                 79: 'toothbrush'}



    def setModelTypeAsCascadeRCNN(self,config_file):
        """
                'setModelTypeAsCascadeRCNN()' is used to set the model type to the Cascade RCNN model
                for the video object detection instance instance object .
                :return:
                """
        self.__modelType = "cascade_rcnn"
        self.config_file = config_file


    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a RetinaNet
          object detection model trained on the COCO dataset.
          :param model_path:
          :return:
        """

        if (self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True

    def loadModel(self, detection_speed="normal"):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function. This function receives an optional value which is "detection_speed".
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.


                * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"

                :param detection_speed:
                :return:
        """

        config_file = self.config_file
        self.__cascade_rcnn_model = init_detector(config_file, self.modelPath, device='cuda')
        cascade_rcnn_cfg = self.__cascade_rcnn_model.cfg
        self.__cascade_rcnn_device = next(self.__cascade_rcnn_model.parameters()).device

        class cascade_rcnn_LoadImage(object):
            def __call__(self, results):
                results['filename'] = None
                img = results['img']
                results['img'] = img
                results['img_shape'] = img.shape
                results['ori_shape'] = img.shape
                return results
        self.__cascade_rcnn_test_pipeline = [cascade_rcnn_LoadImage()] + cascade_rcnn_cfg.data.test.pipeline[1:]
        self.__cascade_rcnn_test_pipeline = Compose(self.__cascade_rcnn_test_pipeline)

        self.__modelLoaded = True



    def CustomObjects(self, person=False, bicycle=False, car=False, motorcycle=False, airplane=False,
                      bus=False, train=False, truck=False, boat=False, traffic_light=False, fire_hydrant=False,
                      stop_sign=False,
                      parking_meter=False, bench=False, bird=False, cat=False, dog=False, horse=False, sheep=False,
                      cow=False, elephant=False, bear=False, zebra=False,
                      giraffe=False, backpack=False, umbrella=False, handbag=False, tie=False, suitcase=False,
                      frisbee=False, skis=False, snowboard=False,
                      sports_ball=False, kite=False, baseball_bat=False, baseball_glove=False, skateboard=False,
                      surfboard=False, tennis_racket=False,
                      bottle=False, wine_glass=False, cup=False, fork=False, knife=False, spoon=False, bowl=False,
                      banana=False, apple=False, sandwich=False, orange=False,
                      broccoli=False, carrot=False, hot_dog=False, pizza=False, donut=False, cake=False, chair=False,
                      couch=False, potted_plant=False, bed=False,
                      dining_table=False, toilet=False, tv=False, laptop=False, mouse=False, remote=False,
                      keyboard=False, cell_phone=False, microwave=False,
                      oven=False, toaster=False, sink=False, refrigerator=False, book=False, clock=False, vase=False,
                      scissors=False, teddy_bear=False, hair_dryer=False,
                      toothbrush=False):

        """
                         The 'CustomObjects()' function allows you to handpick the type of objects you want to detect
                         from an image. The objects are pre-initiated in the function variables and predefined as 'False',
                         which you can easily set to true for any number of objects available.  This function
                         returns a dictionary which must be parsed into the 'detectCustomObjectsFromImage()'. Detecting
                          custom objects only happens when you call the function 'detectCustomObjectsFromImage()'


                        * true_values_of_objects (array); Acceptable values are 'True' and False  for all object values present

                        :param boolean_values:
                        :return: custom_objects_dict
                """

        custom_objects_dict = {}
        input_values = [person, bicycle, car, motorcycle, airplane,
                        bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
                        parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
                        giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
                        sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
                        bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
                        broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed,
                        dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
                        oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
                        toothbrush]
        actual_labels = ["person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

        for input_value, actual_label in zip(input_values, actual_labels):
            if (input_value == True):
                custom_objects_dict[actual_label] = "valid"
            else:
                custom_objects_dict[actual_label] = "invalid"

        return custom_objects_dict

    def detectCustomObjectsFromImage(self, custom_objects=None, input_image="", minimum_percentage_probability=50):
        """
                    'detectCustomObjectsFromImage()' function is used to detect predefined objects observable in the given image path:
                            * custom_objects , an instance of the CustomObject class to filter which objects to detect
                            * input_image , which can be file to path, image numpy array or image file stream
                            * output_image_path , file path to the output image that will contain the detection boxes and label, if output_type="file"
                            * input_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file", "array" and "stream"
                            * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
                            * extract_detected_objects (optional, False by default) , option to save each object detected individually as an image and return an array of the objects' image path.
                            * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                            * display_percentage_probability (optional, True by default), option to show or hide the percentage probability of each object in the saved/returned detected image
                            * display_display_object_name (optional, True by default), option to show or hide the name of each object in the saved/returned detected image
                            * thread_safe (optional, False by default), enforce the loaded detection model works across all threads if set to true, made possible by forcing all Tensorflow inference to run on the default graph.

                    The values returned by this function depends on the parameters parsed. The possible values returnable
            are stated as below
            - If extract_detected_objects = False or at its default value and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = False or at its default value and output_type = 'array' ,
              Then the function will return:

                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = True and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                2. an array of string paths to the image of each object extracted from the image

            - If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                3. an array of numpy arrays of each object detected in the image


            :param input_image:
            :param output_image_path:
            :param input_type:
            :param output_type:
            :param extract_detected_objects:
            :param minimum_percentage_probability:
            :return output_objects_array:
            :param display_percentage_probability:
            :param display_object_name
            :return detected_copy:
            :return detected_detected_objects_image_array:
                """

        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        elif (self.__modelLoaded == True):

            output_objects_array = []
            img_data = input_image
            data = dict(img=img_data)
            data = self.__cascade_rcnn_test_pipeline(data)
            data = scatter(collate([data], samples_per_gpu=1), [self.__cascade_rcnn_device])[0]

            # data['img'][0] = data['img'][0].half()
            # torch.cuda.empty_cache()
            with torch.no_grad():
                result = self.__cascade_rcnn_model(return_loss=False, rescale=True, **data)
            bbox_result, _ = result
            ###
            min_probability = minimum_percentage_probability / 100
            ###
            bboxes = np.vstack(bbox_result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(bbox_result)]
            labels = np.concatenate(labels)
            ####

            for label, detection in zip(labels, bboxes):
                score = detection[4]
                if score < min_probability:
                    continue
                if (custom_objects != None):
                    check_name = self.numbers_to_names[label]
                    if (custom_objects[check_name] == "invalid"):
                        continue
                detection_details = detection[:4].astype(int)
                each_object_details = {}
                each_object_details["name"] = self.numbers_to_names[label]
                each_object_details["percentage_probability"] = score * 100
                each_object_details["box_points"] = detection_details.tolist()
                each_object_details["label"] = label
                output_objects_array.append(each_object_details)

        return output_objects_array
