import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np

def get_gt_boxes(gt_path):
    """Parses MOT format gt.txt file: frame, id, x, y, w, h, ..."""
    boxes_by_frame = {}
    if not os.path.exists(gt_path):
        return boxes_by_frame
    
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame = int(parts[0])
                # MOT format: x, y is top-left
                x, y, w, h = map(float, parts[2:6])
                if frame not in boxes_by_frame:
                    boxes_by_frame[frame] = []
                boxes_by_frame[frame].append((int(x), int(y), int(w), int(h)))
    return boxes_by_frame

def blur_head(img, box):
    """Blurs a tighter region at the top of the bounding box with smooth feathered edges."""
    bx, by, bw, bh = box
    img_h, img_w = img.shape[:2]
    
    hx = max(0, int(bx + bw * 0.1))
    hy = max(0, int(by + bh * 0.02))
    hw = int(bw * 0.8)
    hh = int(bh * 0.18)
    
    hx = min(img_w - 1, hx)
    hy = min(img_h - 1, hy)
    hw = min(img_w - hx, hw)
    hh = min(img_h - hy, hh)
    
    if hw > 3 and hh > 3:
        roi = img[hy:hy+hh, hx:hx+hw].copy()
        
        # Robust blur kernel
        kw = (hw // 2) | 1
        kh = (hh // 2) | 1
        if kw < 3: kw = 3
        if kh < 3: kh = 3

        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 10)
        
        mask = np.zeros((hh, hw), dtype=np.float32)
        cv2.ellipse(mask, (hw // 2, hh // 2), (hw // 2, hh // 2), 0, 0, 360, 1.0, -1)
        
        mask = cv2.GaussianBlur(mask, (kw, kh), 5)
        
        mask = mask[:, :, np.newaxis]
        
        blended = (img[hy:hy+hh, hx:hx+hw] * (1 - mask) + blurred_roi * mask).astype(np.uint8)
        img[hy:hy+hh, hx:hx+hw] = blended
        
    return img

def process_dataset(src_root, dst_root):
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    all_images = []
    for root, dirs, files in os.walk(src_root):
        if dst_root in root: continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_images.append((root, file))

    print(f"Found {len(all_images)} images. Processing using GT knowledge...")
    
    gt_cache = {}

    for root, file in tqdm(all_images, desc="Blurring Faces"):
        rel_path = os.path.relpath(root, src_root)
        dest_dir = os.path.join(dst_root, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        src_file = os.path.join(root, file)
        dst_file = os.path.join(dest_dir, file)

        img = cv2.imread(src_file)
        if img is None:
            shutil.copy2(src_file, dst_file)
            continue

        sequence_dir = os.path.dirname(root) 
        gt_path = os.path.join(sequence_dir, 'gt', 'gt.txt')
        
        if gt_path not in gt_cache:
            gt_cache[gt_path] = get_gt_boxes(gt_path)
        
        boxes = gt_cache[gt_path]
        
        try:

            frame_num = int(''.join(filter(str.isdigit, file)))
        except:
            frame_num = -1

        if frame_num in boxes:

            for box in boxes[frame_num]:
                img = blur_head(img, box)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in detected:
                roi = img[y:y+h, x:x+w]
                kw, kh = (w//2)|1, (h//2)|1
                img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (max(3,kw), max(3,kh)), 30)

        cv2.imwrite(dst_file, img)

    print("Copying remaining files...")
    for root, dirs, files in os.walk(src_root):
        if dst_root in root: continue
        rel_path = os.path.relpath(root, src_root)
        dest_dir = os.path.join(dst_root, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.py')):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    SOURCE_DIRECTORY = r"d:Tracker Dataset"
    DESTINATION_DIRECTORY = r"Tracker Dataset Blurred"
    
    abs_src = os.path.abspath(SOURCE_DIRECTORY)
    abs_dst = os.path.abspath(DESTINATION_DIRECTORY)
    if abs_dst == abs_src or abs_dst.startswith(abs_src + os.sep):
        print("Error: Destination cannot be inside Source.")
        exit(1)

    process_dataset(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)
    print(f"\nDone! Blurred dataset at: {DESTINATION_DIRECTORY}")
