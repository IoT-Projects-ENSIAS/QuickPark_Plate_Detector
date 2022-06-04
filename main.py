from plate_detector import PlateDetector
from plate_ocr import PlateOCR
import cv2
import numpy
import flask
from flask import request, send_file

app = flask.Flask(__name__)
app.config["DEBUG"] = False

plate_model = PlateDetector()
ocr_model = PlateOCR()

@app.route('/detect', methods=['POST'])
def detect():
    plate_ocr = get_plate_ocr(request)
    return plate_ocr['plate_string']

def get_plate_ocr(request):
    image = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_COLOR)
    image = image[:,:,::-1].copy()
    output = plate_model.predict(image)
    plate_boxes = plate_model.plateBoxes(output)
    plates = ocr_model.plateBoxesLoader(image,plate_boxes)
    ocr_strings = []
    for plate in plates:
        ocr_output = ocr_model.predict(plate)
        character_boxes = ocr_model.characterBoxes(ocr_output)
        ocr_strings.append(ocr_model.postProcess(plate,character_boxes))
    return ocr_strings[0]

app.run()