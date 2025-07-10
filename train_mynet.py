from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("light_array.yaml")  # build a new model from scratch; change the path to your model config file

    # Use the model
    model.train(data="light_array_data.yaml", epochs=20, batch=6, imgsz=640, workers=4, device=0)  # train the model; change the path to your data config file
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("00024.jpg")  # predict on an image; change the path to your image
    # path = model.export(format="onnx")  # export the model to ONNX format