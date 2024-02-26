import json

antonio = False

if antonio:
    pos1 = {'x': -7., 'y': 22., 'z': 30.}
    pos2 = {'x': 7., 'y': 22., 'z': 30.}
    camera = {
         "width": 640,
         "height": 480,
        "face_height" : 25.,
        "reference_face_pixel_height": 480.,
        "reference_depth" : 19.,
    }
else:
    pos1 = {'x': -7., 'y': 25., 'z': 33.5}
    pos2 = {'x': 7., 'y': 25., 'z': 33.5}
    camera = {
        "width": 1920,
        "height": 1080,
        "face_height" : 25.,
        "reference_face_pixel_height": 1080.,
        "reference_depth" : 24.,
    }

with open('config.json', 'w') as fp:
        json.dump({
            'arduino_pos': [pos1, pos2],
            'camera': camera,
        }, fp)