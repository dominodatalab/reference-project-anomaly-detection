import sys

from anomalib.deploy import OpenVINOInferencer

inferencer = OpenVINOInferencer(
    path="padim/mvtec/mteal_nut/run/weights/openvino/model.bin",
    metadata="padim/mvtec/mteal_nut/run/weights/openvino/metadata.json",  
    device="CPU"
)

def score(image):
    print("Generating predictions for image {}".format(image))
    predictions = inferencer.predict(image=image)
    return predictions.pred_score, predictions.pred_label

if __name__ == "__main__":
    pred_score, pred_label = score("/domino/datasets/local/anomalib/metal_nut/test/bent/000.png")
    print("pred_score: ", pred_score)
    print("pred_label: ", pred_label)
    
