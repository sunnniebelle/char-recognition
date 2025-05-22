import cv2
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

def segment_characters_horizontal_grouping(image_path, line_threshold=17, space_threshold=20):
    def group_by_lines(boxes, y_thresh):
        lines = []
        boxes = sorted(boxes, key=lambda b: b[1])
        for box in boxes:
            x, y, w, h = box
            placed = False
            for line in lines:
                if abs(y - line[0][1]) < y_thresh:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])
        return lines

    def merge_horizontal_in_line(line_boxes):
        line_boxes = sorted(line_boxes, key=lambda b: b[0])
        merged = []
        current = line_boxes[0]
        for next_box in line_boxes[1:]:
            x1, y1, w1, h1 = current
            x2, y2, w2, h2 = next_box
            if x2 <= x1 + w1 + 2:
                x_new = min(x1, x2)
                y_new = min(y1, y2)
                w_new = max(x1 + w1, x2 + w2) - x_new
                h_new = max(y1 + h1, y2 + h2) - y_new
                current = (x_new, y_new, w_new, h_new)
            else:
                merged.append(current)
                current = next_box
        merged.append(current)
        return merged

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sobel_map = sobel(thresh)

    markers = np.zeros_like(thresh)
    markers[thresh < 30] = 1
    markers[thresh > 150] = 2

    segmentation = watershed(sobel_map, markers)
    normalized = cv2.normalize(segmentation, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    contours, hierarchy = cv2.findContours(normalized, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            approx = cv2.approxPolyDP(c, 3, True)
            boxes.append(cv2.boundingRect(approx))

    lines = group_by_lines(boxes, line_threshold)

    cropped_symbols_with_spaces = []
    for line in lines:
        merged = merge_horizontal_in_line(line)
        merged = sorted(merged, key=lambda b: b[0])
        prev_x = None
        for x, y, w, h in merged:
            if prev_x is not None and (x - prev_x) > space_threshold:
                cropped_symbols_with_spaces.append(" ")
            prev_x = x + w

            x_pad, y_pad = x - 2, y - 2
            w_pad, h_pad = w + 5, h + 5
            cropped = normalized[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
            cropped_symbols_with_spaces.append(cropped)

    return cropped_symbols_with_spaces

symbols = segment_characters_horizontal_grouping('/content/drive/MyDrive/Colab Notebooks/Char-Recognition/letters2.jpg')

fig, axes = plt.subplots(1, len(symbols), figsize=(2 * len(symbols), 4))
for i, sym in enumerate(symbols):
    ax = axes[i]
    if isinstance(sym, str):
        ax.set_title(" ")
        ax.axis('off')
    else:
        ax.imshow(sym, cmap='gray')
        ax.axis('off')
plt.tight_layout()
plt.show()
