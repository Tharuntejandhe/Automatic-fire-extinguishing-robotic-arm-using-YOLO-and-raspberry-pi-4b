import csv



import os



import platform



import sys



from pathlib import Path



import torch



FILE = Path(__file__).resolve()



ROOT = FILE.parents[0] # YOLOv5 root directory



if str(ROOT) not in sys.path:



 sys.path.append(str(ROOT)) # add ROOT to PATH



ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative



from ultralytics.utils.plotting import Annotator, colors, save_one_box



from models.common import DetectMultiBackend



from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams



from utils.general import (



 LOGGER,



 Profile,



 check_file,



 check_img_size,



 check_imshow,



 check_requirements,



 colorstr,



 cv2,



 increment_path,



 non_max_suppression,



 print_args,



 scale_boxes,



 strip_optimizer,



 xyxy2xywh,



)



from utils.torch_utils import select_device, smart_inference_mode



# Function to write detection results to CSV file



def write_to_csv(image_name, prediction, confidence, csv_path):



 """Writes prediction data for an image to a CSV file, appending if the file exists."""



 data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}



 with open(csv_path, mode="a", newline="") as f:



  writer = csv.DictWriter(f, fieldnames=data.keys())



  if not os.path.isfile(csv_path):



   writer.writeheader() # Write header if file is new



  writer.writerow(data)



@smart_inference_mode()



def run(



 weights=ROOT / "../model/yolov5s_best.pt", # model path or triton URL



 source=ROOT / "../test.jpg", # file/dir/URL/glob/screen/0(webcam)



 data=ROOT / "data/coco128.yaml", # dataset.yaml path



 imgsz=(640, 640), # inference size (height, width)



 conf_thres=0.25, # confidence threshold



 iou_thres=0.45, # NMS IOU threshold



 max_det=1000, # maximum detections per image



 device="", # cuda device, i.e. 0 or 0,1,2,3 or cpu



 view_img=False, # show results



 save_txt=False, # save results to *.txt



 save_format=0, # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)



 save_csv=False, # save results in CSV format



 save_conf=False, # save confidences in --save-txt labels



 save_crop=False, # save cropped prediction boxes



 nosave=False, # do not save images/videos



 classes=None, # filter by class: --class 0, or --class 0 2 3



 agnostic_nms=False, # class-agnostic NMS



 augment=False, # augmented inference



 visualize=False, # visualize features



 update=False, # update all models



 project=ROOT / "runs/detect", # save results to project/name



 name="exp", # save results to project/name



 exist_ok=False, # existing project/name ok, do not increment



 line_thickness=3, # bounding box thickness (pixels)



 hide_labels=False, # hide labels



 hide_conf=False, # hide confidences



 half=False, # use FP16 half-precision inference



 dnn=False, # use OpenCV DNN for ONNX inference



 vid_stride=1, # video frame-rate stride



):



 # Same as the original run function implementation...



 source = str(source)



 save_img = not nosave and not source.endswith(".txt") # save inference images



 is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)



 is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))



 webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)



 screenshot = source.lower().startswith("screen")



 if is_url and is_file:



  source = check_file(source) # download



 # Directories



 save_dir = increment_path(Path(project) / name, exist_ok=exist_ok) # increment run



 (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True) # make dir



 # Load model



 device = select_device(device)



 model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)



 stride, names, pt = model.stride, model.names, model.pt



 imgsz = check_img_size(imgsz, s=stride) # check image size



 # Dataloader



 bs = 1 # batch_size



 if webcam:



  view_img = check_imshow(warn=True)



  dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)



  bs = len(dataset)



 elif screenshot:



  dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)



 else:



  dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)



 vid_path, vid_writer = [None] * bs, [None] * bs



 # Run inference



 model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz)) # warmup



 seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))



 detection_vectors = [] # List to store detection vectors (e.g., [cls, conf, xmin, ymin, xmax, ymax])



 for path, im, im0s, vid_cap, s in dataset:



  with dt[0]:



   im = torch.from_numpy(im).to(model.device)



   im = im.half() if model.fp16 else im.float() # uint8 to fp16/32



   im /= 255 # 0 - 255 to 0.0 - 1.0



   if len(im.shape) == 3:



    im = im[None] # expand for batch dim



   if model.xml and im.shape[0] > 1:



    ims = torch.chunk(im, im.shape[0], 0)



  # Inference



  with dt[1]:



   visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False



   if model.xml and im.shape[0] > 1:



    pred = None



    for image in ims:



     if pred is None:



      pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)



     else:



      pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)



    pred = [pred, None]



   else:



    pred = model(im, augment=augment, visualize=visualize)



  # NMS



  with dt[2]:



   pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)



  # Define the path for the CSV file



  csv_path = save_dir / "predictions.csv"



  # Process predictions



  for i, det in enumerate(pred): # per image



   seen += 1



   if webcam: # batch_size >= 1



    p, im0, frame = path[i], im0s[i].copy(), dataset.count



    s += f"{i}: "



   else:



    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)



   p = Path(p) # to Path



   save_path = str(save_dir / p.name) # im.jpg



   txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}") # im.txt



   s += "{:g}x{:g} ".format(*im.shape[2:]) # print string



   gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh



   imc = im0.copy() if save_crop else im0 # for save_crop



   annotator = Annotator(im0, line_width=line_thickness, example=str(names))



   if len(det):



    # Rescale boxes from img_size to im0 size



    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()



    # Print results



    for c in det[:, 5].unique():



     n = (det[:, 5] == c).sum() # detections per class



     s += f"{n} {names[int(c)]}{'s' * (n > 1)} detected, "



    for *xyxy, conf, cls in reversed(det): # xyxy: (xmin, ymin, xmax, ymax), conf: confidence, cls: class index



     # Convert coordinates to integers



     xyxy_int = list(map(int, xyxy)) # Convert each value to integer



     # Create a detection vector: [class, confidence, xmin, ymin, xmax, ymax]



     detection_vector = [int(cls), float(conf)] + xyxy_int



     # Store the detection vector



     detection_vectors.append(detection_vector)



     if save_txt: # Save to text file



      with open(f"{txt_path}.txt", "a") as f:



       f.write(f"{cls} {conf} {' '.join(map(str, xyxy_int))}\n")



     if save_csv: # Save to CSV



      write_to_csv(p.name, names[int(cls)], conf, csv_path)



     if save_img: # Add bbox to image



      label = None if hide_labels else (f"{names[int(cls)]} {conf:.2f}" if not hide_conf else names[int(cls)])



      annotator.box_label(xyxy_int, label, color=colors(int(cls), True))



    # Stream results



    if view_img:



     cv2.imshow(str(p), im0)



     cv2.waitKey(1) # 1 millisecond



   # Save results (image)



   if save_img:



    if dataset.mode == "image":



     cv2.imwrite(save_path, im0)



    else: # video



     if vid_path[i] != save_path: # new video



      vid_path[i] = save_path



      if isinstance(vid_writer[i], cv2.VideoWriter):



       vid_writer[i].release() # release previous video writer



      if vid_cap:



       fps = vid_cap.get(cv2.CAP_PROP_FPS) # video frames per second



       w, h = im0.shape[1], im0.shape[0] # video width, height



       vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))



     vid_writer[i].write(im0)



  LOGGER.info(f"{s}Done. ({dt[0].dt * 1E3:.1f}ms)")



 # Output: List of all detection vectors



 ans=detection_vectors



 print(ans)



 return ans# Print the vector list



 if save_txt or save_csv:



  LOGGER.info(f'Results saved to {save_dir}')



if __name__ == "__main__":



 ans=run(



  weights=ROOT / "../model/yolov5s_best.pt",



  source=ROOT / "../test.jpg",



  data=ROOT / "data/coco128.yaml",



  imgsz=(640, 640),



  conf_thres=0.25,



  iou_thres=0.45,



  max_det=1000,



  device="",



  view_img=False,



  save_txt=True,



  save_format=0,



  save_csv=True,



  save_conf=False,



  save_crop=False,



  nosave=False,



  classes=None,



  agnostic_nms=False,



  augment=False,



  visualize=False,



  update=False,



  project=ROOT / "runs/detect",



  name="exp",



  exist_ok=False,



  line_thickness=3,



  hide_labels=False,



  hide_conf=False,



  half=False,



  dnn=False,



  vid_stride=1,



 )



 print(ans)