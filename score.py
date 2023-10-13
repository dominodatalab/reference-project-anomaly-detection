import os
import sys

from anomalib.deploy import OpenVINOInferencer

# Please change these paths to match where you have stored the model and make sure all workspace artifacts have been syned before starting the model API
inferencer = OpenVINOInferencer(
    path="/mnt/artifacts/padim/mvtec/mteal_nut/run/weights/openvino/model.bin",
    metadata="/mnt/artifacts/padim/mvtec/mteal_nut/run/weights/openvino/metadata.json",  
    device="CPU"
)

def score(image):
    print("Generating predictions for image {}".format(image))
    predictions = inferencer.predict(image=image)
    return predictions.pred_score, predictions.pred_label

# This won't be executed when the model API is run/setup. Also note that Domino datasets are not accessible from Model APIs
if __name__ == "__main__":
    dataset_path = os.path.join(os.environ["DOMINO_DATASETS_DIR"], os.environ["DOMINO_PROJECT_NAME"])
    pred_score, pred_label = score(os.path.join(dataset_path, "metal_nut/test/bent/000.png"))
    print("pred_score: ", pred_score)
    print("pred_label: ", pred_label)
    
