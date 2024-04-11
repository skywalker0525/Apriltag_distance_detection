# from ApriltagDetection import ApriltagDetector
#
# def main():
#     detector = ApriltagDetector()
#     image_path = "corrected_AprilTag_images/40.JPG"
#     detector.detect_tags_in_image(image_path)
#
# if __name__ == '__main__':
#     main()

from ApriltagDetection import ApriltagDetector
import os


def main():
    detector = ApriltagDetector()
    folder_path = "corrected_AprilTag_images_large_angle" 


    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        # test the file type
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")
            detector.detect_tags_in_image(file_path)
        else:
            print(f"Skipping {filename}, not an image or directory.")


if __name__ == '__main__':
    main()

# (0,0) -----------------> x
#   |       p1 * (x1,y1)
#   |            .
#   |                .
#   |                   * p2 (x2,y2)
#   |
#   v
#   y
