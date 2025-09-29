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

OBJ_THRESH = 0.4
NMS_THRESH = 0.4
IMG_SIZE = (640, 640)  # (width, height)

CLASSES = ("plane drone",)

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
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

def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2

    return xyxy

def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i]))
        scores.append(input_data[i][:,4:5,:,:])
        classes_conf.append(input_data[i][:,5:,:,:])

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

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    if boxes is not None:
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

def process_frame(model, frame, co_helper, anchors, platform):
    """Process a single frame for object detection"""
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
    boxes, classes, scores = post_process(outputs, anchors)

    # Draw results
    if boxes is not None:
        draw(img_src, co_helper.get_real_box(boxes), scores, classes)
    
    return img_src, outputs

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Performance Benchmark')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    parser.add_argument('--video_path', type=str, required=True, help='Path to video file for testing')
    parser.add_argument('--anchors', type=str, default='../model/anchors_yolov5.txt', help='anchor file path')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup frames')
    parser.add_argument('--test_frames', type=int, default=2000, help='Number of frames to test')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()

    # Load anchors
    with open(args.anchors, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3, -1, 2).tolist()
    print("Using anchors from '{}'".format(args.anchors))

    # Initialize model
    model, platform = setup_model(args)
    co_helper = COCO_test_helper(enable_letter_box=True)

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        model.release()
        return
    
    # Performance measurement variables
    processing_times = []
    inference_times = []
    postprocess_times = []
    frame_count = 0
    
    print(f"Starting performance benchmark...")
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
        
        process_frame(model, frame, co_helper, anchors, platform)
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
            
            # Measure inference time only
            img_src = frame.copy()
            pad_color = (0, 0, 0)
            img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if platform in ['pytorch', 'onnx']:
                input_data = img.transpose((2, 0, 1))
                input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
                input_data = input_data / 255.
            else:
                input_data = img

            # Inference time measurement
            inference_start = time.time()
            outputs = model.run([input_data])
            inference_time = time.time() - inference_start
            
            # Post-processing time measurement
            postprocess_start = time.time()
            boxes, classes, scores = post_process(outputs, anchors)
            postprocess_time = time.time() - postprocess_start

            # Draw results
            if boxes is not None:
                draw(img_src, co_helper.get_real_box(boxes), scores, classes)
            
            total_time = time.time() - frame_start
            
            # Store timings
            processing_times.append(total_time)
            inference_times.append(inference_time)
            postprocess_times.append(postprocess_time)
            
            frame_count += 1
            
            # Display with performance info
            if not args.no_display:
                current_fps = 1.0 / total_time if total_time > 0 else 0
                cv2.putText(img_src, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_src, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_src, f"Post-process: {postprocess_time*1000:.1f}ms", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_src, f"Frame: {frame_count}/{args.test_frames}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('YOLOv5 Performance Benchmark', img_src)
                
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
        print("YOLOv5 PERFORMANCE BENCHMARK RESULTS")
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
