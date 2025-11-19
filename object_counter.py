import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


class ObjectCounter:

    def __init__(self, min_area=500, blur_kernel=5, threshold_block=11):
        self.min_area = min_area
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.threshold_block = threshold_block if threshold_block % 2 == 1 else threshold_block + 1

        self.color_ranges = {
            'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                    (np.array([170, 100, 100]), np.array([180, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
            'orange': [(np.array([10, 100, 100]), np.array([25, 255, 255]))]
        }

    def preprocess_image(self, image, target_size=(800, 600)):
        height, width = image.shape[:2]
        if width > target_size[0] or height > target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        return image

    def count_objects_threshold(self, image_path, show_steps=False):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        image = self.preprocess_image(image)
        original = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.threshold_block, 2
        )

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        result_image = self.draw_contours(original, filtered_contours)

        if show_steps:
            self._show_processing_steps(image, gray, blurred, thresh, opened, result_image)

        return len(filtered_contours), result_image, filtered_contours

    def count_objects_color(self, image_path, color='red', show_steps=False):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        image = self.preprocess_image(image)
        original = image.copy()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color.lower() not in self.color_ranges:
            raise ValueError(f"Color '{color}' not supported. Choose from: {list(self.color_ranges.keys())}")

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges[color.lower()]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        result_image = self.draw_contours(original, filtered_contours, color=color)

        if show_steps:
            self._show_color_processing_steps(image, hsv, mask, result_image)

        return len(filtered_contours), result_image, mask

    def draw_contours(self, image, contours, color=None):
        result = image.copy()

        box_color = (0, 255, 0)
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 2)

            cv2.drawContours(result, [contour], -1, box_color, 2)

            label = f"#{i+1}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(result,
                         (x, y - label_size[1] - 10),
                         (x + label_size[0] + 5, y),
                         bg_color, -1)

            cv2.putText(result, label, (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        count_text = f"Total Count: {len(contours)}"
        if color:
            count_text = f"Total {color.capitalize()} Objects: {len(contours)}"

        count_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(result, (10, 10), (count_size[0] + 30, 60), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (count_size[0] + 30, 60), (0, 255, 0), 3)

        cv2.putText(result, count_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        return result

    def _show_processing_steps(self, original, gray, blurred, thresh, opened, result):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Object Counting - Processing Steps (Threshold Method)', fontsize=16)

        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('2. Grayscale')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(blurred, cmap='gray')
        axes[0, 2].set_title('3. Gaussian Blur')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(thresh, cmap='gray')
        axes[1, 0].set_title('4. Adaptive Threshold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(opened, cmap='gray')
        axes[1, 1].set_title('5. Morphological Operations')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('6. Final Result with Count')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def _show_color_processing_steps(self, original, hsv, mask, result):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Object Counting - Processing Steps (Color Method)', fontsize=16)

        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        axes[0, 1].set_title('2. HSV Color Space')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('3. Color Mask')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('4. Final Result with Count')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def save_result(self, image, output_path):
        cv2.imwrite(output_path, image)
        print(f"Result saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Smart Object Counter - Count objects in images using OpenCV',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--image', '-i', required=True,
                       help='Path to input image')
    parser.add_argument('--method', '-m', choices=['threshold', 'color'],
                       default='threshold',
                       help='Counting method to use (default: threshold)')
    parser.add_argument('--color', '-c',
                       choices=['red', 'green', 'blue', 'yellow', 'orange'],
                       default='red',
                       help='Color to detect (only for color method)')
    parser.add_argument('--min-area', type=int, default=500,
                       help='Minimum contour area (default: 500)')
    parser.add_argument('--output', '-o',
                       help='Output path for annotated image')
    parser.add_argument('--show-steps', action='store_true',
                       help='Show intermediate processing steps')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    counter = ObjectCounter(min_area=args.min_area)

    print(f"\nProcessing image: {args.image}")
    print(f"Method: {args.method}")

    try:
        if args.method == 'threshold':
            count, result_image, _ = counter.count_objects_threshold(
                args.image, show_steps=args.show_steps
            )
        else:
            print(f"Color: {args.color}")
            count, result_image, _ = counter.count_objects_color(
                args.image, color=args.color, show_steps=args.show_steps
            )

        print(f"\n{'='*50}")
        print(f"RESULT: {count} objects detected!")
        print(f"{'='*50}\n")

        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.image)
            output_dir = Path('results')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_counted{input_path.suffix}"

        counter.save_result(result_image, str(output_path))

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Object Counting Result: {count} objects')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
