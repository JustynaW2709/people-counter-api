from flask import Flask
from flask_restful import Resource, Api
import cv2

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/dworzec.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))

        return {'count': len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


# class HelloWorld2(Resource):
#   def get(self):
#       return {'hello': 'world2'}

# endpoint pierwszy
api.add_resource(HelloWorld, '/test')
# api.add_resource(HelloWorld2, '/test2') endpoint drugi
api.add_resource(PeopleCounter, '/')

if __name__ == '__main__':
    app.run(debug=True)
