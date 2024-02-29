from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

#class HelloWorld2(Resource):
#   def get(self):
#       return {'hello': 'world2'}

api.add_resource(HelloWorld, '/test') #endpoint pierwszy
#api.add_resource(HelloWorld2, '/test2') endpoint drugi

if __name__ == '__main__':
    app.run(debug=True)