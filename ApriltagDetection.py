# $-----------------------------$
#
# File Name: ApriltagDetection.py
#
# $-----------------------------$

import apriltag
import cv2
import numpy as np
import VisionException


class ApriltagDetector:

    def __init__(self) -> None:
        pass  # 删除了与视频流相关的初始化逻辑

    def __createDetector(self) -> apriltag.Detector:
        """
        Creates an apriltag detector that is made for tag36h11 tags and returns it.
        """
        return apriltag.Detector(apriltag.DetectorOptions(families='tag36h11',
                                                      border=1,
                                                      nthreads=4,
                                                      quad_decimate=1,
                                                      quad_blur=2,  # 注意这里的改动
                                                      refine_edges=True,
                                                      refine_decode=False,
                                                      refine_pose=False,
                                                      debug=False,
                                                      quad_contours=True))  # 增加解码前的锐化)

    def __findDistance(self, objectHeight, objectWidth) -> float:
        """
        Takes the object's height and width in pixels and then calculates distance to the target in mm. Mind that dependent on camera model and target used the constants
        declared will vary.
        """
        REAL_OBJECT_HEIGHT_AND_WIDTH_IN = 25.5  # mm
        FOCAL_DISTANCE_CONSTANT = 4057.241520467836   # mm x 3587.80081 y 3590.25677

        # Prevent division by zero by checking if objectHeight or objectWidth is zero
        if objectHeight == 0 or objectWidth == 0:
            print("Warning: Object height or width is zero, cannot calculate distance.")
            return -1  # Or handle this scenario in a way that fits your application

        # You'll want to take the larger of the two values as the larger will be the most accurate to the target if it's at an angle.
        distance = (REAL_OBJECT_HEIGHT_AND_WIDTH_IN * FOCAL_DISTANCE_CONSTANT) / max(objectHeight, objectWidth)/(10)

        return distance

    def __whenApriltagDetected(self, apriltagsDetected, image) -> np.ndarray:
        """
        Function which is called whenever an apriltag is detected.
        """
        print("[INFO] {} Apriltags Detected.".format(len(apriltagsDetected)))
        return self.__drawAroundApriltags(apriltagsDetected, image)

    def preprocess_underwater_image(self, image) -> np.ndarray:
        # 转换到YUV色彩空间并进行直方图均衡化
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

        # 应用CLAHE增加对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])

        # 将处理后的图像转换回BGR色彩空间
        equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        # 应用双边滤波减少噪声，同时尽可能保持边缘
        filtered_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)  # 使用适中的双边滤波参数

        # cv2.imshow('Processed Image', filtered_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return filtered_image

    def detect_tags_in_image(self, image_path: str) -> None:
        """
        Detects AprilTags in a given image file and annotates the image with the detections.
        """
        # 读取图片
        image = cv2.imread(image_path)
        image = self.preprocess_underwater_image(image)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return

        # 将图片转换为灰度
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 创建检测器并检测AprilTags
        detector = self.__createDetector()
        tags = detector.detect(gray_image)

        # 标记检测到的tags并计算它们的信息
        if tags:
            annotated_image = self.__drawAroundApriltags(tags, image)  # 使用原图来绘制边界和ID
            # 构造保存路径
            save_path = image_path.rsplit('.', 1)
            save_path = f"{save_path[0]}_detected.{save_path[1]}"
            cv2.imwrite(save_path, annotated_image)
            print(f"Annotated image saved as {save_path}")
        else:
            print("No AprilTags detected.")

    def __drawAroundApriltags(self, apriltags, image) -> np.ndarray:
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        tag_counter = 1  # 初始化计数器
        for tag in apriltags:
            # 绘制边界框
            (ptA, ptB, ptC, ptD) = tag.corners
            cv2.polylines(image, [np.int32(tag.corners)], True, (0, 255, 0), 2)

            # 计算Tag的中心坐标
            center = tag.center
            tag_id = tag.tag_id

            # 在图像上绘制Tag的ID和顺序
            cv2.putText(image, f"ID: {tag_id} ({tag_counter})",
                        (int(center[0]) - 10, int(center[1]) - 10), FONT, 0.5, (0, 255, 0), 2)

            # 计算Tag在图像上的大小（以像素为单位）和距离
            objectHeight_px = np.linalg.norm(ptA - ptB)  # 像素单位的高度
            objectWidth_px = np.linalg.norm(ptB - ptC)  # 像素单位的宽度
            size_px = max(objectHeight_px, objectWidth_px)  # 使用较大的尺寸作为P
            distance_mm = self.__findDistance(objectHeight_px, objectWidth_px)  # 假设返回值是以毫米为单位

            # 在控制台打印Tag的全部信息（包括中心坐标、尺寸、距离和顺序）
            print(
                f"AprilTag #{tag_counter} ID: {tag_id}, Center: ({int(center[0])}, {int(center[1])}), Size: {round(size_px, 2)}px, Distance: {round(distance_mm, 2)}cm")

            tag_counter += 1  # 更新计数器

        return image

    def __showVideo(self, image) -> None:
        # Opens the window and then waits for a keypress to stop.
        cv2.imshow("Apriltag Detection", image)

    def __startDetection(self, videoStream) -> None:
        """
        Infinite loop which loops through the frames in the video to find apriltags and then output to console the number found.
        """
        while (True):
            # Analyze the video stream frame-by-frame.
            ret, frame = videoStream.read()

            # Assures that the frame has been read correctly. If not then an exception is thrown.
            if not ret:
                raise VisionException.FrameNotFound()

            # Changes the image to a single-channel (black and white).
            grayedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Outputs the number of apriltags detected.
            apriltagDetector = self.__createDetector()
            detectedTags = apriltagDetector.detect(grayedFrame)
            numDetected = len(detectedTags)

            if numDetected != 0:
                self.__showVideo(self.__whenApriltagDetected(detectedTags, frame))
            else:
                print("[INFO] No Apriltags Detected")
                self.__showVideo(frame)

            cv2.waitKey(1)