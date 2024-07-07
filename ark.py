from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np
import pytesseract


# FILENAME = 'ark_test_rotated.png'
FILENAME = 'ark_test.png'


class ImageParseError(Exception):
   pass


@dataclass
class PreprocessedImage:
  orig: cv2.typing.MatLike
  gray: cv2.typing.MatLike
  blur: cv2.typing.MatLike
  invert: cv2.typing.MatLike


def preprocess_img(img: cv2.typing.MatLike) -> PreprocessedImage:
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite('tmp/gray.png', gray)
  blur = cv2.GaussianBlur(gray, (7, 7), 0)
  cv2.imwrite('tmp/blur.png', blur)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  cv2.imwrite('tmp/thresh.png', thresh)
  return PreprocessedImage(img, gray, blur, thresh)


def get_bounding_boxes(img: PreprocessedImage, kernal=(13, 3), min_width: Optional[int] = 240) -> list[cv2.typing.Rect]:
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernal)
  dilate = cv2.dilate(img.invert, kernel, iterations=1)
  cv2.imwrite('tmp/dilate.png', dilate)
  contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Remove small/overlapping bounding boxes
  bounding_boxes = [
    cv2.boundingRect(contour)
    for contour in contours
    if cv2.boundingRect(contour)[2] > min_width
  ]

  # Sort from top down
  return sorted(bounding_boxes, key=lambda bb: bb[1])


def draw_bounding_boxes(img: PreprocessedImage, bounding_boxes: list[cv2.typing.Rect]) -> cv2.typing.MatLike:
  orig = img.orig.copy()
  for bb in bounding_boxes:
    x, y, w, h = bb
    cv2.rectangle(orig, (x, y), (x+w, y+h), (36, 255, 12), 2)
  draw_image(orig)
  return img


def crop_bounding_boxes(img: PreprocessedImage, bounding_boxes: list[cv2.typing.Rect]) -> list[cv2.typing.MatLike]:
  orig = img.orig.copy()
  cropped_imgs = []
  for bb in bounding_boxes:
    x, y, w, h = bb
    cropped_img = orig[y:y+h, x:x+w]
    cropped_imgs.append(cropped_img)
  return cropped_imgs


def draw_image(img: cv2.typing.MatLike):
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyWindow('image')


def crop_rows(img: PreprocessedImage, padding=5) -> list[cv2.typing.MatLike]:
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
  erode = cv2.erode(img.invert, kernel, iterations=2)
  canny = cv2.Canny(erode, 0, 150)

  # cv.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) → lines
  rho = 1
  theta = np.pi/180
  threshold = 50
  minLinLength = 150
  maxLineGap = 6
  lines = cv2.HoughLinesP(canny, rho, theta, threshold, None, minLinLength, maxLineGap)

  if lines is None:
    print('No row lines found')
    return []

  sorted_lines: list[cv2.typing.MatLike] = sorted(lines, key=lambda l: l[0][1])
  bucketed_lines = bucket_lines(sorted_lines, sorting_index=1)

  # Draw lines on a blank image
  blank_image = np.ones(img.orig.shape, dtype=np.uint8)
  for line in bucketed_lines:
    _, y1, _, y2 = line
    cv2.line(blank_image, (0, y1), (img.orig.shape[1], y2), (255,255,255), 3, cv2.LINE_AA)

  # Get bounding boxes from drawn lines
  preprocessed = preprocess_img(blank_image)
  bounding_boxes = get_bounding_boxes(preprocessed)
  # img_sans_lines = cv2.add(img.orig, preprocessed.orig)

  # Crop bounding boxes and return
  rows = []
  for bb in bounding_boxes:
    x, y, w, h = bb
    top_padding = max(0, y - padding)
    bottom_padding = min(y + h + padding, img.orig.shape[0])
    cropped_img = img.orig[top_padding:bottom_padding, x:x+w]
    rows.append(cropped_img)

  return rows


def crop_cols(img: PreprocessedImage, padding=5) -> list[cv2.typing.MatLike]:
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
  erode = cv2.erode(img.invert, kernel, iterations=1)
  canny = cv2.Canny(erode, 0, 50)

  # cv.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) → lines
  rho = 1
  theta = np.pi/360
  threshold = 20
  minLinLength = img.orig.shape[0]*.6
  maxLineGap = 6
  lines = cv2.HoughLinesP(canny, rho, theta, threshold, None, minLinLength, maxLineGap)

  if lines is None:
    print('No column lines found')
    return []

  sorted_lines: list[cv2.typing.MatLike] = sorted(lines, key=lambda l: l[0][1])
  bucketed_lines = bucket_lines(sorted_lines, sorting_index=0)

  # Draw lines on a blank image
  blank_image = np.ones(img.orig.shape, dtype=np.uint8)
  for line in bucketed_lines:
    x1, _, x2, _ = line
    cv2.line(blank_image, (x1, 0), (x2, img.orig.shape[0]), (255,255,255), 3, cv2.LINE_AA)

  # Get bounding boxes from drawn lines
  preprocessed = preprocess_img(blank_image)
  bounding_boxes = get_bounding_boxes(preprocessed, kernal=(3, 13), min_width=5)
  bounding_boxes = sorted(bounding_boxes, key=lambda bb: bb[0])
  # img_sans_lines = cv2.add(img.orig, preprocessed.orig)

  # Crop bounding boxes and return
  cols = []
  for bb in bounding_boxes:
    x, y, w, h = bb
    left_padding = max(0, x - padding)
    right_padding = min(x + w + padding, img.orig.shape[1])
    cropped_img = img.orig[y:y+h, left_padding:right_padding]
    cols.append(cropped_img)

  return cols


def bucket_lines(lines: list[cv2.typing.MatLike], sorting_index: Optional[Union[0, 1]]=1) -> list[cv2.typing.MatLike]:
  buckets = {}
  buckted_lines = []
  i = 0

  while i < len(lines):
    neighbors = set()
    # Look ahead
    j = i
    while j < len(lines):
      k = j + 1
      curr = lines[j][0]
      neighbors.add(curr[sorting_index])
      j = k
      if not k < len(lines):
        break
      next_line = lines[k][0]
      avg = sum(neighbors) // len(neighbors)
      if avg + 5 < next_line[sorting_index]:
        break
      neighbors.add(next_line[sorting_index])

    i = j

    avg = sum(neighbors) // len(neighbors)
    for neighbor in neighbors:
      buckets[neighbor] = avg

  for line in lines:
    x, y, w, h = line[0]
    new_y = buckets[y] if sorting_index else buckets[x]
    buckted_lines.append((x, new_y, w, h))

  return buckted_lines


def draw_table_lines(img: PreprocessedImage) -> cv2.typing.MatLike:
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
  erode = cv2.erode(img.invert, kernel, iterations=1)
  draw_image(erode)
  canny = cv2.Canny(erode, 50, 150)
  # canny = cv2.Canny(blur, 50, 150)
  # draw_image(canny)

  # cv.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) → lines
  rho = 1
  theta = np.pi/180
  threshold = 50
  minLinLength = 350
  maxLineGap = 1
  lines = cv2.HoughLinesP(canny, rho , theta, threshold, None, minLinLength, maxLineGap)

  if lines is None:
    raise ImageParseError

  horizontal_lines = {}
  vertical_lines = {}

  for i in range(0, len(lines)):
    l = lines[i][0]
    x, y, w, h = l
    if is_vertical(l):
      if x in vertical_lines:
        vertical_lines[x][1] = min(y, vertical_lines[x][1])
        vertical_lines[x][3] = max(h, vertical_lines[x][3])
      else:
        vertical_lines[x] = [x, y, w, h]
    elif is_horizontal(l):
      if y in horizontal_lines:
        horizontal_lines[y][0] = min(x, horizontal_lines[y][0])
        horizontal_lines[y][2] = max(w, horizontal_lines[y][2], horizontal_lines[y][0] + w)
      else:
        horizontal_lines[y] = [x, y, w, h]

  horizontal_lines = [
    (x, y, w, h)
    for x, y, w, h in horizontal_lines.values()
    if w > 800
  ]

  vertical_lines = [
    (x, y, w, h)
    for x, y, w, h in vertical_lines.values()
  ]

  print(len(horizontal_lines))
  for _, line in enumerate(horizontal_lines):
    x, y, w, h = line
    cv2.line(img, (x, y), (w, h), (0,0,255), 3, cv2.LINE_AA)
  for _, line in enumerate(vertical_lines):
    x, y, w, h = line
    cv2.line(img, (x, y), (w, h), (0,255,0), 3, cv2.LINE_AA)

  draw_image(img)
  return img


def is_vertical(line):
    return line[0]==line[2]


def is_horizontal(line):
    return line[1]==line[3]


def extract_text(img: cv2.typing.MatLike) -> str:
  img = preprocess_img(img)
  # Remove lines
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
  erode = cv2.erode(img.invert, kernel, iterations=2)
  canny = cv2.Canny(erode, 0, 150)
  print('*'*80)
  draw_image(img.orig)
  draw_image(erode)
  without_lines = cv2.add(img.orig, erode)
  draw_image(without_lines)

  # cv.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) → lines
  rho = 1
  theta = np.pi/180
  threshold = 50
  minLinLength = 150
  maxLineGap = 6
  lines = cv2.HoughLinesP(canny, rho, theta, threshold, None, minLinLength, maxLineGap)

  if lines:
    print(lines)
    # Draw lines on a blank image
    blank_image = np.ones(img.orig.shape, dtype=np.uint8)
    for line in lines:
      x1, _, x2, _ = line
      cv2.line(blank_image, (x1, 0), (x2, img.orig.shape[0]), (255,255,255), 3, cv2.LINE_AA)
    draw_image(blank_image)

  return

  # Dilate image
  kernel_to_remove_gaps_between_words = np.array([
          # [1,1,1,1,1,1,1,1,1,1],
          # [1,1,1,1,1,1,1,1,1,1]
          [4,4,4,4],
          [4,4,4,4],
  ])
  dilate = cv2.dilate(img.invert, kernel_to_remove_gaps_between_words, iterations=2)
  draw_image(img.orig)
  draw_image(dilate)

  # Find contours
  contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  img_w_contours = img.orig.copy()
  cv2.drawContours(img_w_contours, contours, -1, (0, 255, 0), 3)
  draw_image(img_w_contours)

  # Convert contours to bounding boxes
  bounding_boxes = [
    cv2.boundingRect(contour)
    for contour in contours
    if cv2.boundingRect(contour)[2] > 25
  ]
  bounding_boxes = sorted(bounding_boxes, key=lambda bb: bb[1])
  img_with_bounding_boxes = img.orig.copy()

  all_text = []
  for bb in bounding_boxes:
    x, y, w, h = bb
    padding_top = y - 5
    padding_bottom = y + h + 5
    padding_left = x - 5
    padding_right = x + w + 5
    cropped = img.orig[padding_top:padding_bottom, padding_left:padding_right]
    text = pytesseract.image_to_string(cropped, config='--psm 3 --oem 1')
    # print(text)
    all_text.append(text)
    img_with_bounding_boxes = cv2.rectangle(img_with_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), )
    draw_image(cropped)
  # draw_image(img_with_bounding_boxes)

  return '\n'.join(all_text)


def tesseract_extract(img: cv2.typing.MatLike) -> str:
  return pytesseract.image_to_string(img, config='--psm 3 --dpi 36 --oem 1')


def main():
  img = cv2.imread(cv2.samples.findFile(FILENAME))
  img = preprocess_img(img)
  bounding_boxes = get_bounding_boxes(img)
  cropped_imgs = crop_bounding_boxes(img, bounding_boxes)
  for cropped_img in cropped_imgs[2:]:
    cropped_img = preprocess_img(cropped_img)
    rows = crop_rows(cropped_img)
    for row in rows:
      preprocessed_row = preprocess_img(row)
      cols = crop_cols(preprocessed_row)
      for col in cols:
        # text = extract_text(col)
        text = tesseract_extract(col)
        if text:
          print(text.strip().replace('\n', ' '), end=',')
        else:
          print('', end=',')
        # draw_image(col)

      print('\n')
      # draw_image(row)


if __name__ == '__main__':
  main()
