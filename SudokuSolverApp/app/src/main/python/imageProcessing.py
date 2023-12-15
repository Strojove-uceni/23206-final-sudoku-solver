import cv2
import numpy as np
from PIL import Image
import base64
import io
import tflite_runtime.interpreter as tflite
from os.path import dirname, join
from sudoku import Sudoku

def generate_image(board):
  def get_image_as_array(number):
    if number == None or number == 0:
      return np.invert(np.zeros((100,100), dtype = np.uint8))

    image = np.ones((100, 100, 1), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_thickness = 3
    cv2.putText(image, str(number), (10, 90), font, font_scale, (0, 0, 0), font_thickness)

    return np.squeeze(image,2)


  def get_image_with_border(number):
    x = get_image_as_array(number)
    b = 5
    return cv2.copyMakeBorder(x, b,b,b,b, cv2.BORDER_CONSTANT, value=[122])

  def make_row(row):
    sequence = []
    for i in board[row]:
      sequence.append(get_image_with_border(i))
    return np.hstack(tuple(sequence))

  def make_board():
    sequence = []
    for i in range(9):
      sequence.append(make_row(i))
    return np.vstack(tuple(sequence))

  return make_board()

def read_sudoku_from_image(image):
  mean = 139.445
  std = 37.7531
  
  #load digit classification model
  model_path = join(dirname(__file__), "digitCls.tflite")
  interpreter = tflite.Interpreter(model_path=model_path)
  
  interpreter.allocate_tensors()
  inputIndex = interpreter.get_input_details()[0]["index"]
  outputIndex = interpreter.get_output_details()[0]["index"]

  #transform into greyscale
  sudoku_image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sudoku_image = sudoku_image_grayscale.copy()

  #find contours
  cv2.GaussianBlur(sudoku_image, (9, 9), 0, sudoku_image)
  cv2.adaptiveThreshold(sudoku_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, sudoku_image)
  contours = cv2.findContours(sudoku_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  #there are no contours, no sudoku found -> return empty string
  if len(contours) == 0:
    return ""

  contours = sorted(contours, key=cv2.contourArea, reverse=True) #sort contours to find the largest one
  largest_contour = contours[0]

  corners1 = largest_contour[:4].copy()
  corners2 = largest_contour[:4].copy()

  #find corners of the sudoku
  #we search for 2 squares and than check which square has larger area -> works also for rotated images
  for pt in largest_contour:
    if pt[0][0] + pt[0][1] < corners1[0][0][0] + corners1[0][0][1]: corners1[0] = pt
    if pt[0][0] - pt[0][1] > corners1[1][0][0] - corners1[1][0][1]: corners1[1] = pt
    if pt[0][0] + pt[0][1] > corners1[2][0][0] + corners1[2][0][1]: corners1[2] = pt
    if pt[0][0] - pt[0][1] < corners1[3][0][0] - corners1[3][0][1]: corners1[3] = pt
    if pt[0][0] < corners2[0][0][0]: corners2[0] = pt
    if pt[0][1] < corners2[1][0][1]: corners2[1] = pt
    if pt[0][0] > corners2[2][0][0]: corners2[2] = pt
    if pt[0][1] > corners2[3][0][1]: corners2[3] = pt

  #compare the areas
  corners = np.empty(0)
  if cv2.contourArea(corners1) > cv2.contourArea(corners2):
    corners = corners1.squeeze()
  else:
    corners = corners2.squeeze()

  grid_side = max([np.sqrt(p[0]*p[0] + p[1]*p[1]) for p in corners]) #find longes distance between corners

  #transform the sudoku image to a square
  corners_new = np.array([[0, 0], [grid_side - 1, 0], [grid_side - 1, grid_side - 1], [0, grid_side - 1]], dtype="float32")
  transform = cv2.getPerspectiveTransform(corners.astype("float32"), corners_new)
  sudoku_image_grayscale = cv2.warpPerspective(sudoku_image_grayscale, transform, (int(grid_side), int(grid_side)))

  #cut the square sudoku image into images of individual cells and recognize digits in cells
  output = ""
  cell_side = grid_side / 9
  for i in range(9):
    for j in range(9):
      top_left = (int(i * cell_side), int(j * cell_side))
      bottom_right = (int((i + 1) * cell_side), int((j + 1) * cell_side))
      number_image = sudoku_image_grayscale[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
      number_image = cv2.resize(number_image, (28, 28))
      number_image = (number_image - mean) / std
      model_input = number_image.reshape((1, 1, 28, 28)).astype(np.float32)
      interpreter.set_tensor(inputIndex, model_input)
      interpreter.invoke()
      pred = str(interpreter.get_tensor(outputIndex).argmax())
      output += pred
  return output #return recognized digits as a string


def main(imageString):
    decoded_image = base64.b64decode(imageString)
    np_image = np.fromstring(decoded_image, np.uint8)
    sudoku_image_rgb = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)
    sudoku_string = read_sudoku_from_image(sudoku_image_rgb)

    if sudoku_string == "":
      image_path = "failed.jpg"
      pil_image = Image.open(image_path)
      buff = io.BytesIO()
      pil_image.save(buff, format="PNG")
      img_str = base64.b64encode(buff.getvalue())

      return ""+str(img_str, "utf-8")
      
    sudoku_array = [list(map(int, sudoku_string[i:i+9])) for i in range(0, 81, 9)]
    sudoku_puzzle = Sudoku(3, 3, board=sudoku_array)

    if not sudoku_puzzle.validate():
      image_path = join(dirname(__file__), "failed.jpg")
      pil_image = Image.open(image_path)
      buff = io.BytesIO()
      pil_image.save(buff, format="PNG")
      img_str = base64.b64encode(buff.getvalue())

      return ""+str(img_str, "utf-8")
    else:
      solution = sudoku_puzzle.solve()
      result_image = generate_image(solution.board)
      pil_image = Image.fromarray(result_image)
      buff = io.BytesIO()
      pil_image.save(buff, format="PNG")
      img_str = base64.b64encode(buff.getvalue())

      return ""+str(img_str, "utf-8")

"""def main(imageString):
    decoded_image = base64.b64decode(imageString)
    np_image = np.fromstring(decoded_image, np.uint8)
    sudoku_image_rgb = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)
    sudoku_image = cv2.cvtColor(sudoku_image_rgb, cv2.COLOR_BGR2GRAY)

    cv2.GaussianBlur(sudoku_image, (9, 9), 0, sudoku_image)
    cv2.adaptiveThreshold(sudoku_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, sudoku_image)

    contours = cv2.findContours(sudoku_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True) #sort contours to fing the largest one
    largest_contour = contours[0]

    corners1 = largest_contour[:4].copy()
    corners2 = largest_contour[:4].copy()

    #find corners of the sudoku
    #we search for 2 squares and than check which square has larger area -> works also for rotated images
    for pt in largest_contour:
        if pt[0][0] + pt[0][1] < corners1[0][0][0] + corners1[0][0][1]: corners1[0] = pt
        if pt[0][0] - pt[0][1] > corners1[1][0][0] - corners1[1][0][1]: corners1[1] = pt
        if pt[0][0] + pt[0][1] > corners1[2][0][0] + corners1[2][0][1]: corners1[2] = pt
        if pt[0][0] - pt[0][1] < corners1[3][0][0] - corners1[3][0][1]: corners1[3] = pt
        if pt[0][0] < corners2[0][0][0]: corners2[0] = pt
        if pt[0][1] < corners2[1][0][1]: corners2[1] = pt
        if pt[0][0] > corners2[2][0][0]: corners2[2] = pt
        if pt[0][1] > corners2[3][0][1]: corners2[3] = pt

    if cv2.contourArea(corners1) > cv2.contourArea(corners2):
        cv2.drawContours(sudoku_image_rgb, corners1, -1, (0,255,0), 15)
    else:
        cv2.drawContours(sudoku_image_rgb, corners2, -1, (0,255,0), 15)

    pil_image = Image.fromarray(sudoku_image_rgb)
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())

    return ""+str(img_str, "utf-8")"""
