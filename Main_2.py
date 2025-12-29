from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
from Pipeline_2Fix(1) import predict_from_video

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    # Create temp directory if doesn't exist
    os.makedirs("temp_videos", exist_ok=True)
    
    temp_path = f"temp_videos/{file.filename}"
    
    try:
        # Save uploaded file
        logger.info(f"Saving file to {temp_path}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            logger.info(f"File saved, size: {len(content)} bytes")
        
        # Check file exists and has content
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise HTTPException(status_code=400, detail="File upload failed or file is empty")
        
        logger.info("Starting video analysis...")

        result = predict_from_video(temp_path)
        logger.info(f"Analysis complete. Result: {result}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Could not remove temp file: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

