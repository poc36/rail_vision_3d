"""
Calibrate — Интерактивная калибровка стерео-камеры по шахматной доске.
"""

import cv2
import numpy as np
import argparse
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Stereo Camera Calibration")
    parser.add_argument("--left", default=0, help="Left camera ID or video path")
    parser.add_argument("--right", default=1, help="Right camera ID or video path")
    parser.add_argument("--cols", type=int, default=9, help="Chessboard inner corners (cols)")
    parser.add_argument("--rows", type=int, default=6, help="Chessboard inner corners (rows)")
    parser.add_argument("--square-size", type=float, default=25.0, help="Square size in mm")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames to capture")
    parser.add_argument("--output", default="data/calibration/stereo_calib", help="Output path")
    args = parser.parse_args()

    from src.camera.stereo_camera import StereoCamera
    from src.camera.calibration import StereoCalibrator

    # Определить тип источника
    try:
        left_src = int(args.left)
        right_src = int(args.right)
    except ValueError:
        left_src = args.left
        right_src = args.right

    # Открыть камеры
    camera = StereoCamera(left_source=left_src, right_source=right_src)
    if not camera.open():
        logger.error("Cannot open cameras")
        return

    calibrator = StereoCalibrator(
        chessboard_size=(args.cols, args.rows),
        square_size_mm=args.square_size,
    )

    left_images = []
    right_images = []
    captured = 0

    logger.info(f"Calibration: capturing {args.num_frames} pairs")
    logger.info("Press SPACE to capture, Q to finish early")

    while captured < args.num_frames:
        ret, left, right = camera.read()
        if not ret:
            break

        # Показать текущий кадр
        display = np.hstack([left, right])

        # Попробовать найти шахматную доску
        found_l, corners_l = calibrator.find_chessboard_corners(left)
        found_r, corners_r = calibrator.find_chessboard_corners(right)

        show_left = left.copy()
        show_right = right.copy()

        if found_l:
            cv2.drawChessboardCorners(
                show_left, (args.cols, args.rows), corners_l, found_l
            )
        if found_r:
            cv2.drawChessboardCorners(
                show_right, (args.cols, args.rows), corners_r, found_r
            )

        display = np.hstack([show_left, show_right])

        status = f"Captured: {captured}/{args.num_frames}"
        if found_l and found_r:
            status += " [DETECTED - Press SPACE]"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            status += " [Searching...]"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stereo Calibration", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and found_l and found_r:
            left_images.append(left.copy())
            right_images.append(right.copy())
            captured += 1
            logger.info(f"Captured pair {captured}")

    camera.release()
    cv2.destroyAllWindows()

    if captured < 5:
        logger.error(f"Not enough pairs captured: {captured}. Need at least 5.")
        return

    # Калибровка
    logger.info(f"Calibrating with {captured} pairs...")
    rms = calibrator.calibrate(left_images, right_images)
    logger.info(f"Calibration RMS error: {rms:.4f}")

    # Сохранение
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    calibrator.save(args.output)
    logger.info(f"Calibration saved to {args.output}")

    # Тест ректификации
    if len(left_images) > 0:
        rect_l, rect_r = calibrator.rectify(left_images[0], right_images[0])
        display = np.hstack([rect_l, rect_r])
        # Нарисовать горизонтальные линии для проверки ректификации
        for y in range(0, display.shape[0], 30):
            cv2.line(display, (0, y), (display.shape[1], y), (0, 255, 0), 1)
        cv2.imshow("Rectified (check horizontal lines)", display)
        logger.info("Showing rectified image. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
