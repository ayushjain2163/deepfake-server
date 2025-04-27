from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import base64
import uuid
import os
import logging

import features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def home():
    return JSONResponse(content='API is running')

@app.post("/predictVideo")
async def predict_video(video: UploadFile = File(...)):

    try:
        video_path = "video.mp4"
        content = await video.read()
        with open(video_path, "wb") as video_file:
            video_file.write(content)

        prediction = features.video_classifier(video_path)
        return JSONResponse(content={'result':prediction})
    
    except:
        return JSONResponse(content={"message":"Error in reading Video Data"})
    

@app.post("/predictImage")
async def predict_image(image: UploadFile = File(...)):

    try:
        image_path = 'image.jpg'
        content = await image.read()
        with open(image_path, "wb") as video_file:
            video_file.write(content)
        
        prediction = features.image_classifier(image_path)
        return JSONResponse(content={'result':prediction})

    except:
        return JSONResponse(content={"message":"Error in reading Image Data"})



@app.post("/extensionPredict")
async def extension_predict(data: dict = Body(...)):
    try:
        base64_images = data.get("images", [])
        result = "real"

        for img_str in base64_images:
            # Remove data URL prefix if present
            if "base64," in img_str:
                img_str = img_str.split("base64,")[1]

            try:
                # Decode base64 string
                img_data = base64.b64decode(img_str)
            except Exception as e:
                return JSONResponse(
                    content={"message": f"Invalid base64 image: {str(e)}"},
                    status_code=400
                )

            # Create temporary file
            temp_file = f"{uuid.uuid4()}.jpg"
            with open(temp_file, "wb") as f:
                f.write(img_data)

            # Use existing prediction logic
            prediction = features.image_classifier(temp_file)
            print("Prediction:", prediction)
            os.remove(temp_file)  # Clean up immediately
            if prediction==1:
                result = "fake"
                break  # No need to check remaining images

        return JSONResponse(content={"result": result})

    except Exception as e:
        logging.exception("Error processing images")
        return JSONResponse(
            content={"message": "Error processing images"},
            status_code=500
        )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=4000)