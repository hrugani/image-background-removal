import sys
import datetime
import os
from io import BytesIO
import numpy as np
import boto3
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """
        Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """

        start = datetime.datetime.now()

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))

        return resized_image, seg_map


def get_s3_client():
    """ """
    my_key = os.getenv('KEY')
    my_secret = os.getenv('SECRET')
    """get the boto3.client object with the AWS security access rules and policies applied """
    s3 = boto3.client('s3', aws_access_key_id=my_key, aws_secret_access_key=my_secret)
    return s3


def download_input_file():
    """ """
    try:
        s3 = get_s3_client()
        with open(INPUT_FILE_NAME, 'wb') as f:
            s3.download_fileobj(INPUT_BUCKET_NAME, INPUT_FILE_NAME, f)
    except Exception as e:
        raise e


def upload_output_file():
    """save in S3 the new image with white background"""
    try:
        s3 = get_s3_client()
        with open(OUTPUT_FILE_NAME, 'rb') as f:
            s3.upload_fileobj(f, OUTPUT_BUCKET_NAME, OUTPUT_FILE_NAME)
    except Exception as e:
        raise e


def generate_no_background(model):
    """Inferences DeepLab model and generate image without background."""
    try:
        print("Trying to open : " + INPUT_FILE_NAME)
        jpeg_str = open(INPUT_FILE_NAME, "rb").read()
        original_im = Image.open(BytesIO(jpeg_str))
        print('running deeplab on image %s...' % INPUT_FILE_NAME)
        resized_im, seg_map = model.run(original_im)
        new_image = create_new_image(resized_im, seg_map)
        new_image.save(OUTPUT_FILE_NAME)
    except Exception as e:
        raise e


def create_new_image(base_img, mat_img):
    width, height = base_img.size
    dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = mat_img[y, x]
            (r, g, b) = base_img.getpixel((x, y))
            if color == 0:
                dummy_img[y, x, 3] = 0
            else:
                dummy_img[y, x] = [r, g, b, 255]
    return Image.fromarray(dummy_img)


def clear_local_area():
    try:
        os.remove(INPUT_FILE_NAME)
        os.remove(OUTPUT_FILE_NAME)
    except Exception as e:
        raise e


def create_output_file_name(input_file_name):
    name, ext = os.path.splitext(input_file_name)
    return name + "-no-background.png"


INPUT_BUCKET_NAME = 'dev-sptrans-photos-original'            # must be got from "aws s3 create object event"
INPUT_FILE_NAME = "dne-sample.jpg"                           # must be got from "aws s3 create object event"
OUTPUT_BUCKET_NAME = 'dev-sptrans-photos-no-background'      # constant - project decision
OUTPUT_FILE_NAME = create_output_file_name(INPUT_FILE_NAME)  # based in INPUT_FILE_NAME
MODEL_TYPE_DESC = "xception_model"  # use "xception_model" for major quality, or "mobile_net_model" for draft quality


def main():

    print('loading model: ', MODEL_TYPE_DESC)
    model = DeepLabModel(MODEL_TYPE_DESC)
    print('model loaded successfully : ' + MODEL_TYPE_DESC)

    download_input_file()
    print("[OK] image:", INPUT_FILE_NAME, "downloaded")
    generate_no_background(model)
    print("[OK] image:", OUTPUT_FILE_NAME, "generated")
    upload_output_file()
    print("[OK] image:", OUTPUT_FILE_NAME, "Uploaded")
    clear_local_area()
    print("[OK] tmp files:", INPUT_FILE_NAME, OUTPUT_FILE_NAME, "deleted")


if __name__ == '__main__':
    main()

##########################################################################
# Auxiliary functions and classes - bellow
##########################################################################




