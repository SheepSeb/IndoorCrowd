from ultralytics.models.sam import SAM3VideoSemanticPredictor

# Initialize semantic video predictor
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=640,
    model="sam3.pt",
    half=True,
    save=True,
)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

source = "/home/sebnae/shared_drive/ws/yet_another_person_tracker/dataset/raw_video/train/ie_recording_2026-02-24_17-36-23.mp4"

results = predictor(source=source, text=["person"], stream=True)
for result in results:
    result.show()
