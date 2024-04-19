import cv2
import numpy as np
import apriltag
import matplotlib.pyplot as plt

class ApriltagDetector:
    def __init__(self):
        self.detector = self.__createDetector()
        self.tag_sizes = {
            0: 8.5,   # Size for tag ID 0 in mm
            2: 17.2,  # Size for tag ID 2 in mm
            10: 25.5, # Size for tag ID 10 in mm
            12: 38.5  # Size for tag ID 12 in mm
        }

    def __createDetector(self):
        options = apriltag.DetectorOptions(families='tag36h11',
                                           border=1,
                                           nthreads=4,
                                           quad_decimate=1.0,
                                           quad_blur=2.0,
                                           refine_edges=True,
                                           refine_decode=False,
                                           refine_pose=False,
                                           debug=False,
                                           quad_contours=True)
        return apriltag.Detector(options)

    def __findDistance(self, objectHeight, objectWidth, tag_id):
        FOCAL_DISTANCE_CONSTANT = 4057.241520467836  # Pre-calibrated focal distance constant
        if objectHeight == 0 or objectWidth == 0:
            return float('inf')  # Avoid division by zero
        max_dimension = max(objectHeight, objectWidth)
        tag_size = self.tag_sizes.get(tag_id, 25.5)  # Default to 25.5 mm if tag ID is not specified
        return (tag_size * FOCAL_DISTANCE_CONSTANT) / max_dimension / 10  # Convert result to cm

    def detect_tags_in_video(self, video_path: str, output_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        tag_data = {id: {'times': [], 'distances': []} for id in self.tag_sizes.keys()}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray_image)
            annotated_image = self.__drawAroundApriltags(tags, frame.copy(), cap, tag_data)
            out.write(annotated_image)
            cv2.imshow('Apriltag Detection', annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved as {output_path}")

        # Plot distance versus time for each tag
        for tag_id, data in tag_data.items():
            if data['times']:
                plt.figure()
                plt.plot(data['times'], data['distances'], marker='o', linestyle='-', label=f'Tag ID {tag_id}')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Distance (cm)')
                plt.title(f'Distance over Time for Tag ID {tag_id}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_path.rsplit('.', 1)[0]}_tag{tag_id}_distance_plot.png")
                plt.close()

    def __drawAroundApriltags(self, apriltags, image, cap, tag_data):
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        for tag in apriltags:
            (ptA, ptB, ptC, ptD) = tag.corners
            cv2.polylines(image, [np.array([ptA, ptB, ptC, ptD], dtype=np.int32)], True, (0, 255, 0), 2)
            center = np.mean([ptA, ptB, ptC, ptD], axis=0)
            objectHeight_px = np.linalg.norm(ptA - ptB)
            objectWidth_px = np.linalg.norm(ptB - ptC)
            distance = self.__findDistance(objectHeight_px, objectWidth_px, tag.tag_id) / 2  # Divided by 2 as specified
            cv2.putText(image, f"ID: {tag.tag_id}, Dist: {distance:.2f} cm",
                        (int(center[0]), int(center[1])), FONT, 0.5, (0, 255, 0), 2)
            tag_data[tag.tag_id]['times'].append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)  # Convert ms to seconds
            tag_data[tag.tag_id]['distances'].append(distance)
            cv2.putText(image, f"ID {tag.tag_id}: {distance:.2f} cm",
                        (image.shape[1] - 200, 30 + 30 * int(tag.tag_id)), FONT, 0.5, (0, 0, 255), 2)
        return image

# Example usage
detector = ApriltagDetector()
# detector.detect_tags_in_video('/home/tiantan/Documents/Apriltag_distance_detection/test/large angle.MP4', 
#                               '/home/tiantan/Documents/Apriltag_distance_detection/test/large angle_output.MP4')

# detector.detect_tags_in_video('/home/tiantan/Documents/Apriltag_distance_detection/test/medium angle.MP4', 
#                               '/home/tiantan/Documents/Apriltag_distance_detection/test/medium angle_output.MP4')

# detector.detect_tags_in_video('/home/tiantan/Documents/Apriltag_distance_detection/test/small angle.MP4', 
#                               '/home/tiantan/Documents/Apriltag_distance_detection/test/small angle_output.MP4')

detector.detect_tags_in_video('/home/tiantan/Documents/Apriltag_distance_detection/test/straight.MP4', 
                              '/home/tiantan/Documents/Apriltag_distance_detection/test/straight_output.MP4')

# detector.detect_tags_in_video('/home/tiantan/Documents/Apriltag_distance_detection/test/wide lens.MP4', 
#                               '/home/tiantan/Documents/Apriltag_distance_detection/test/wide lens_output.MP4')