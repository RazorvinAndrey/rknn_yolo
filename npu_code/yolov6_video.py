import os
import cv2
import sys
import argparse
import time
import numpy as np

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)

CLASSES = ("planedrone",)

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes."""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    if position.shape[1]==4:
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    else:
        position = dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model'.format(model_path, platform))
    return model, platform

def process_frame_yolov6(model, frame, co_helper, platform):
    """Process a single frame for YOLOv6 object detection"""
    img_src = frame.copy()
    
    # Preprocess image
    pad_color = (0, 0, 0)
    img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess for different platforms
    if platform in ['pytorch', 'onnx']:
        input_data = img.transpose((2, 0, 1))
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        input_data = input_data / 255.
    else:
        input_data = img

    # Run inference
    outputs = model.run([input_data])
    boxes, classes, scores = post_process(outputs)

    # Draw results
    if boxes is not None:
        draw(img_src, co_helper.get_real_box(boxes), scores, classes)
    
    return img_src, outputs

def main():
    parser = argparse.ArgumentParser(description='YOLOv6 Performance Benchmark')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    parser.add_argument('--video_path', type=str, required=True, help='Path to video file for testing')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup frames')
    parser.add_argument('--test_frames', type=int, default=100, help='Number of frames to test')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()

    # Initialize model
    model, platform = setup_model(args)
    co_helper = COCO_test_helper(enable_letter_box=True)

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    #cap = cv2.VideoCapture(21)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        model.release()
        return
    
    # Performance measurement variables
    processing_times = []
    inference_times = []
    postprocess_times = []
    frame_count = 0
    
    print(f"Starting YOLOv6 performance benchmark...")
    print(f"Warmup frames: {args.warmup}")
    print(f"Test frames: {args.test_frames}")
    print("Press 'q' to stop early")
    
    # Warmup phase
    print("\n=== Warmup Phase ===")
    for i in range(args.warmup):
        ret, frame = cap.read()
        if not ret:
            print("Not enough frames for warmup!")
            break
        
        process_frame_yolov6(model, frame, co_helper, platform)
        frame_count += 1
        
        if i % 5 == 0:
            print(f"Warmup frame {i+1}/{args.warmup}")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    
    # Main testing phase
    print("\n=== Testing Phase ===")
    start_time = time.time()
    
    try:
        while frame_count < args.test_frames:
            ret, frame = cap.read()
            
            if not ret:
                # Loop video if we reach the end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Measure total processing time
            frame_start = time.time()
            
            # Preprocessing
            img_src = frame.copy()
            pad_color = (0, 0, 0)
            img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
            img = co_helper.letter_box(im=frame, new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if platform in ['pytorch', 'onnx']:
                input_data = img.transpose((2, 0, 1))
                input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
                input_data = input_data / 255.
            else:
                input_data = img
            
            input_data = img
            # Inference time measurement
            inference_start = time.time()
            outputs = model.run([input_data])
            inference_time = time.time() - inference_start
            
            # Post-processing time measurement
            postprocess_start = time.time()
            boxes, classes, scores = post_process(outputs)
            postprocess_time = time.time() - postprocess_start
            
            
            
            total_time = time.time() - frame_start
            
            # Store timings
            processing_times.append(total_time)
            inference_times.append(inference_time)
            postprocess_times.append(postprocess_time)
            
            frame_count += 1
            
            # Display with performance info
            # Draw results
            
            if boxes is not None:
                draw(frame, co_helper.get_real_box(boxes), scores, classes)
                
            if not args.no_display:
                current_fps = 1.0 / total_time if total_time > 0 else 0
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Post-process: {postprocess_time*1000:.1f}ms", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{args.test_frames}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('YOLOv6 Performance Benchmark', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
             
             
            
            # Print progress
            if frame_count % 10 == 0:
                current_avg_fps = frame_count / (time.time() - start_time)
                print(f"Processed {frame_count}/{args.test_frames} frames, "
                      f"Current FPS: {current_avg_fps:.2f}")
                
    finally:
        # Cleanup
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        model.release()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        total_fps = frame_count / total_time
        
        # Convert to milliseconds
        processing_ms = [t * 1000 for t in processing_times]
        inference_ms = [t * 1000 for t in inference_times]
        postprocess_ms = [t * 1000 for t in postprocess_times]
        
        print("\n" + "="*50)
        print("YOLOv6 PERFORMANCE BENCHMARK RESULTS")
        print("="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average FPS: {total_fps:.2f}")
        print(f"Max possible FPS: {1/np.min(processing_times):.2f}")
        print("\n--- Processing Time Statistics ---")
        print(f"Average total processing: {np.mean(processing_ms):.2f}ms")
        print(f"Min total processing: {np.min(processing_ms):.2f}ms")
        print(f"Max total processing: {np.max(processing_ms):.2f}ms")
        print(f"Std dev processing: {np.std(processing_ms):.2f}ms")
        
        print("\n--- Inference Time Statistics ---")
        print(f"Average inference: {np.mean(inference_ms):.2f}ms")
        print(f"Min inference: {np.min(inference_ms):.2f}ms")
        print(f"Max inference: {np.max(inference_ms):.2f}ms")
        print(f"Inference % of total: {np.mean(inference_times)/np.mean(processing_times)*100:.1f}%")
        
        print("\n--- Post-processing Time Statistics ---")
        print(f"Average post-process: {np.mean(postprocess_ms):.2f}ms")
        print(f"Min post-process: {np.min(postprocess_ms):.2f}ms")
        print(f"Max post-process: {np.max(postprocess_ms):.2f}ms")
        print(f"Post-process % of total: {np.mean(postprocess_times)/np.mean(processing_times)*100:.1f}%")

if __name__ == '__main__':
    main()
