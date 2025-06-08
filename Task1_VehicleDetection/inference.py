import cv2
import numpy as np
import logging
import time
import json
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ParkingSpaceDetector:
    """
    Handles parking space detection for both images and videos
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class mapping (fixed to match training)
        self.class_names = {
            0: 'space-empty',
            1: 'space-occupied'
        }
        
        # Colors for visualization
        self.colors = {
            'space-empty': (0, 255, 0),    # Green
            'space-occupied': (0, 0, 255)  # Red
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Load model
        self.model = self.load_model()
        
        logger.info(f"Detector initialized with confidence threshold: {confidence_threshold}")
    
    def load_model(self):
        """Load YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            model = YOLO(str(self.model_path))
            logger.info("✅ Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def detect_parking_spaces(self, image: np.ndarray) -> Dict:
        """
        Detect parking spaces in a single image
        Returns: Dictionary with detection results
        """
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            # Process results
            detection_data = {
                'total_vehicles': 0,
                'empty_spaces': 0,
                'occupied_spaces': 0,
                'detections': [],
                'confidence_scores': []
            }
            
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = self.class_names.get(cls, f'class_{cls}')
                        
                        # Count spaces (fixed logic)
                        if class_name == 'space-empty':
                            detection_data['empty_spaces'] += 1
                        elif class_name == 'space-occupied':
                            detection_data['occupied_spaces'] += 1
                            detection_data['total_vehicles'] += 1
                        
                        # Store detection info
                        detection_data['detections'].append({
                            'bbox': [x1, y1, x2, y2],
                            'class': class_name,
                            'confidence': float(conf)
                        })
                        detection_data['confidence_scores'].append(float(conf))
            
            return detection_data
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {'total_vehicles': 0, 'empty_spaces': 0, 'occupied_spaces': 0, 'detections': []}
    
    def draw_detections(self, image: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        annotated_image = image.copy()
        
        for detection in detection_data['detections']:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Get color
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Label text
            cv2.putText(annotated_image, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 255, 255), 2)
        
        return annotated_image
    
    def add_info_overlay(self, image: np.ndarray, detection_data: Dict, fps: Optional[float] = None) -> np.ndarray:
        """Add information overlay to image"""
        overlay_img = image.copy()
        
        # Info text
        info_lines = [
            f"Total Vehicles: {detection_data['total_vehicles']}",
            f"Empty Spaces: {detection_data['empty_spaces']}",
            f"Occupied Spaces: {detection_data['occupied_spaces']}",
            f"Total Detections: {len(detection_data['detections'])}"
        ]
        
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")
        
        # Calculate occupancy rate
        total_spaces = detection_data['empty_spaces'] + detection_data['occupied_spaces']
        if total_spaces > 0:
            occupancy_rate = (detection_data['occupied_spaces'] / total_spaces) * 100
            info_lines.append(f"Occupancy: {occupancy_rate:.1f}%")
        
        # Draw background
        overlay_height = len(info_lines) * 30 + 20
        cv2.rectangle(overlay_img, (10, 10), (350, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(overlay_img, (10, 10), (350, overlay_height), (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(overlay_img, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay_img
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """Process single image"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect parking spaces
        start_time = time.time()
        detection_data = self.detect_parking_spaces(image)
        inference_time = time.time() - start_time
        
        # Add timing info
        detection_data['inference_time'] = inference_time
        
        # Create annotated image
        annotated_image = self.draw_detections(image, detection_data)
        annotated_image = self.add_info_overlay(annotated_image, detection_data)
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            logger.info(f"Result saved to: {output_path}")
        
        # Log results
        logger.info(f"Detection Results:")
        logger.info(f"  Total Vehicles: {detection_data['total_vehicles']}")
        logger.info(f"  Empty Spaces: {detection_data['empty_spaces']}")
        logger.info(f"  Occupied Spaces: {detection_data['occupied_spaces']}")
        logger.info(f"  Inference Time: {inference_time:.3f}s")
        
        return detection_data
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     display: bool = True) -> Dict:
        """Process video file"""
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        frame_count = 0
        total_inference_time = 0
        avg_detection_data = {
            'total_vehicles': 0,
            'empty_spaces': 0,
            'occupied_spaces': 0
        }
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect parking spaces
                start_time = time.time()
                detection_data = self.detect_parking_spaces(frame)
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # Update averages
                for key in avg_detection_data:
                    avg_detection_data[key] += detection_data[key]
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:  # Calculate FPS every 30 frames
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                else:
                    current_fps = None
                
                # Create annotated frame
                annotated_frame = self.draw_detections(frame, detection_data)
                annotated_frame = self.add_info_overlay(annotated_frame, detection_data, current_fps)
                
                # Write frame if output path provided
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Parking Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processing interrupted by user")
                        break
                
                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        if frame_count > 0:
            for key in avg_detection_data:
                avg_detection_data[key] = avg_detection_data[key] / frame_count
            
            avg_inference_time = total_inference_time / frame_count
            
            # Log final results
            logger.info(f"Video Processing Complete:")
            logger.info(f"  Total Frames Processed: {frame_count}")
            logger.info(f"  Average Inference Time: {avg_inference_time:.3f}s")
            logger.info(f"  Average Vehicles per Frame: {avg_detection_data['total_vehicles']:.1f}")
            logger.info(f"  Average Empty Spaces: {avg_detection_data['empty_spaces']:.1f}")
            logger.info(f"  Average Occupied Spaces: {avg_detection_data['occupied_spaces']:.1f}")
            
            if output_path:
                logger.info(f"Output video saved to: {output_path}")
        
        return {
            'frames_processed': frame_count,
            'avg_inference_time': avg_inference_time if frame_count > 0 else 0,
            'avg_detections': avg_detection_data
        }
    
    def process_realtime_camera(self, camera_id: int = 0, display: bool = True) -> None:
        """Process real-time camera feed"""
        logger.info(f"Starting real-time camera processing (Camera ID: {camera_id})")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        logger.info("Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Detect parking spaces
                detection_data = self.detect_parking_spaces(frame)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Create annotated frame
                annotated_frame = self.draw_detections(frame, detection_data)
                annotated_frame = self.add_info_overlay(annotated_frame, detection_data, current_fps)
                
                # Display frame
                if display:
                    cv2.imshow('Real-time Parking Detection', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"parking_snapshot_{timestamp}.jpg"
                        cv2.imwrite(filename, annotated_frame)
                        logger.info(f"Snapshot saved: {filename}")
        
        except KeyboardInterrupt:
            logger.info("Real-time processing interrupted by user")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Parking Space Detection')
    parser.add_argument('--model', required=True, help='Path to YOLO model')
    parser.add_argument('--source', required=True, help='Source: image path, video path, or camera ID')
    parser.add_argument('--output', help='Output path for results')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], default='image', help='Processing mode')
    parser.add_argument('--no-display', action='store_true', help='Disable display for video/camera mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = ParkingSpaceDetector(args.model, args.conf)
        
        if args.mode == 'image':
            # Process single image
            detector.process_image(args.source, args.output)
        
        elif args.mode == 'video':
            # Process video
            display = not args.no_display
            detector.process_video(args.source, args.output, display)
        
        elif args.mode == 'camera':
            # Process camera feed
            camera_id = int(args.source) if args.source.isdigit() else 0
            display = not args.no_display
            detector.process_realtime_camera(camera_id, display)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    # python parking_detector.py --model /teamspace/studios/this_studio/Task1_VehicleDetection/runs/train/parking_detection/weights/best.pt --source /teamspace/studios/this_studio/test_image.jpg --mode image --output result.jpg
    # python parking_detector.py --model best.pt --source video.mp4 --mode video --output result.mp4
    # python parking_detector.py --model best.pt --source 0 --mode camera
    main()