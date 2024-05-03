from fastapi import FastAPI, File
from fastapi.exceptions import HTTPException
from model import RetinaNet
from utils import from_bytes_to_pil

app = FastAPI()
model = RetinaNet()

@app.get("/ping")
def pong():
    return "pong"

@app.post("/retinanet/predict")
async def get_prediction(input_file: bytes = File(...)):
    '''
    Input: Image file as bytes
    Output: BBoxes and labels
    '''
    try:
        input_image = from_bytes_to_pil(input_file)
    except Exception as exp:
        return HTTPException(status_code=400, detail=f'Image could not be read {str(exp)}')
    
    try:
        predictions = model(input_image)
    except Exception as exp:
        return HTTPException(status_code=500, detail=f'There was an error during the model execution {str(exp)}')

    return predictions