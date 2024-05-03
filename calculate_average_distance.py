import cv2
import numpy as np
import apriltag
import os
from tqdm import tqdm  # Import tqdm for progress bars
import csv

class ApriltagImageDetector:
    def __init__(self):
        self.detector = self.__createDetector()
        self.tag_sizes = {
            0: 8.5,   # Size for tag ID 0 in mm
            2: 17.2,  # Size for tag ID 2 in mm
            10: 25.5, # Size for tag ID 10 in mm
            12: 38.5  # Size for tag ID 12 in mm
        }
        self.FOCAL_DISTANCE_CONSTANT = 934.8987012987013

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
        if objectHeight == 0 or objectWidth == 0:
            return None
        max_dimension = max(objectHeight, objectWidth)
        tag_size = self.tag_sizes.get(tag_id, 25.5)  
        return (tag_size * self.FOCAL_DISTANCE_CONSTANT) / max_dimension / 10  

    def process_images_in_folder(self, folder_path):
        filenames = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
        tag_distances = {id: [] for id in self.tag_sizes.keys()}
        for filename in tqdm(filenames, desc=f"Processing images in {os.path.basename(folder_path)}"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                tags = self.detector.detect(gray_image)
                for tag in tags:
                    (ptA, ptB, ptC, ptD) = tag.corners
                    objectHeight_px = np.linalg.norm(ptA - ptB)
                    objectWidth_px = np.linalg.norm(ptB - ptC)
                    distance = self.__findDistance(objectHeight_px, objectWidth_px, tag.tag_id)
                    if distance is not None:
                        tag_distances[tag.tag_id].append(distance)
        return {tag_id: (np.mean(distances) if distances else None) for tag_id, distances in tag_distances.items()}

    def save_results_to_csv(self, results, csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Category", "Subfolder", "Tag ID", "Average Distance (cm)"])
            for category, subfolders in results.items():
                for subfolder, distances in subfolders.items():
                    for tag_id, distance in distances.items():
                        writer.writerow([category, subfolder, tag_id, distance])

    def process_all_folders(self, base_folder):
        categories = [c for c in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, c))]
        all_results = {}
        for category in tqdm(categories, desc="Processing categories"):
            category_path = os.path.join(base_folder, category)
            category_results = {}
            subfolders = [sf for sf in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, sf))]
            for subfolder in tqdm(subfolders, desc=f"Processing subfolders in {category}"):
                subfolder_path = os.path.join(category_path, subfolder)
                category_results[subfolder] = self.process_images_in_folder(subfolder_path)
            all_results[category] = category_results
        return all_results

# Example usage
detector = ApriltagImageDetector()
base_folder = 'data/4.28_image'  # Adjust the path as needed
results = detector.process_all_folders(base_folder)
csv_path = '4.28_calculated.csv'  # Define the path for the CSV file
detector.save_results_to_csv(results, csv_path)