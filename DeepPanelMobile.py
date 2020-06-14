import tensorflow as tf

if __name__ == "__main__":
    export_dir = "./model/"
    print(f" - Loading model from {export_dir}")
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    print(f" - Starting model conversion")
    tflite_model = converter.convert()
    with tf.io.gfile.GFile(f"{export_dir}deepPanel.tflite", "wb") as f:
        f.write(tflite_model)
    print(f" - Model converted and saved in {export_dir}")
