# Anomaly Detection using Computer Vision

The purpose of anomaly detection via computer vision is to identify and flag instances in visual data that deviate significantly from what is considered normal or expected. Anomalies, also known as outliers, are data points or patterns that differ substantially from the majority of the data, either due to errors, defects, fraud, or any other unusual circumstances. Anomaly detection in computer vision involves leveraging machine learning and image processing techniques to automatically identify such anomalies in images or videos.

In this project, we go through the process of applying anomaly detection for the purposes of quality control and defect detection. We fit a [PaDIM](https://arxiv.org/abs/2011.08785) model against the [MVTecAD](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf) dataset using [Anomalib](https://github.com/openvinotoolkit/anomalib/tree/main), a comprehensive deep learning library designed to serve as a hub for state-of-the-art anomaly detection algorithms. 

The assets included in the project are:

*anomaly_detection.ipynb* - A notebook that walks the user over the process of acquiring the MVTecAD dataset and storing it as a [Domino Dataset](https://docs.dominodatalab.com/en/latest/user_guide/0a8d11/datasets/). It also configures, trains, and runs a test inference call with the PaDIM model. The trained model is also persisted in `padim` for the purposes of deploying it as a Model API.

*score.py* - a scoring function that exposes the persisted model as Model API. The score function accepts an image path as an argument and returns a boolean prediction (anomalous or not) and a confidence score of the prediction.

## Dockerfile

This project uses a compute environment based on quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.2-domino5.6-standard

Add the following entries to the Dockerfile:

```
USER ubuntu
RUN pip install anomalib==0.6.0 \
    && pip install openvino==2023.0.1

#RUN pip install onnx==1.14.0

RUN pip install openvino-dev[pytorch,onnx]==2023.0.1
```

Don't forget to expose the relevant IDEs as pluggable workspaces, as described in the [Domino Documentation](https://docs.dominodatalab.com/en/latest/user_guide/03e062/add-workspace-ides/).

## Model API

The scoring endpoint expects an image path as its input. For this to work, the model version needs to be configured with a a [Kubernetes volumes](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/deploy-models-at-rest/#add-volumes), where the scored images are uploaded. The call then takes the form of

```
{
  "data": {
    "image": "[volume]/[image.png]"
  }
}
```


