import sys
import datetime
import os
from io import BytesIO
import numpy as np
import boto3
from PIL import Image
# from Pillow import Image (I tried this line instead the previous line - no good results - delete this line
# import tensorflow as tf (this import didn't work. This is the original import written by the author)
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
        """Runs inference on a single image.

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


def create_new_image(baseImg, matImg):
    width, height = baseImg.size
    dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = matImg[y, x]
            (r, g, b) = baseImg.getpixel((x, y))
            if color == 0:
                dummy_img[y, x, 3] = 0
            else:
                dummy_img[y, x] = [r, g, b, 255]
    return Image.fromarray(dummy_img)


def remove_background(file_path, model):
    """Inferences DeepLab model and generate image wothout background."""
    try:
        print("Trying to open : " + file_path)
        jpeg_str = open(file_path, "rb").read()
        orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check file: ' + file_path)
        return

    print('running deeplab on image %s...' % file_path)
    resized_im, seg_map = model.run(orignal_im)

    new_image = create_new_image(resized_im, seg_map)
    output_file_name = create_output_file_name(file_path)
    new_image.save(output_file_name)

    return output_file_name


def create_output_file_name(input_file_path):
    file_base_name = os.path.basename(input_file_path)
    pre, ext = os.path.splitext(file_base_name)
    return pre + ".png"


def get_input_file_names(dir_name):
    f_names = os.listdir(dir_name)
    return f_names


def upload_to_s3(local_file_path):
    """Uploads a local file to AWS-S3 bucket that has the name = "output-png-images\"
    Uses AWS_ACCESS_KEY from an environment variable with the same name
    Uses AWS_SECRET_ACCESS_KEY from another environment variable with the same name
    """
    key_id = os.getenv('AWS_ACCESS_KEY_ID')
    secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3_bucket_name = 'output-png-images'
    s3_file_name = create_output_file_name(local_file_path)
    s3 = boto3.client('s3', aws_access_key_id=key_id, aws_secret_access_key=secret)
    try:
        with open(local_file_path, 'rb') as data:
            s3.upload_fileobj(data, s3_bucket_name, s3_file_name)
        print("[OK] image:", local_file_path, "Uploaded")
        return True
    except FileNotFoundError as e:
        print("[ERROR] The file was not found : ", local_file_path)
        return False
    except IOError:
        print('[ERROR] I/O error when uploading file:', local_file_path)
        return False
    except Exception as e:
        print("[ERROR]", "When uploading file: [", local_file_path, "]", e)
        return False

###########################################################################
# main - begin the main routine
###########################################################################


def main():

    if len(sys.argv) != 2:
        print("parameter error. Please specify the model_type parameter (0 ==> mobile_net_model, 1 ==> xception_model)")
        exit()

    model_type = sys.argv[1]

    if model_type is None:
        print("Bad parameter. Please specify the model_type (0 = mobile_net_model, 1 = xception_model)")
        exit()
    if model_type != '0' and model_type != '1':
        print("Bad parameter value. Allowed values [0 or 1]")
        exit()

    if model_type == '1':
        model_type_desc = "xception_model"
    else:
        model_type_desc = "mobile_net_model"

    print('loading model: ', model_type_desc)
    model = DeepLabModel(model_type_desc)
    print('model loaded successfully : ' + model_type_desc)

    input_dir_name = "input-jpg-images"
    input_file_names = get_input_file_names(input_dir_name)
    for file_name in input_file_names:
        file_path = input_dir_name + "/" + file_name
        output_file_name = remove_background(file_path, model)
        upload_to_s3(output_file_name)
        os.remove(output_file_name)

###########################################################################
# main - end
###########################################################################


if __name__ == '__main__':
    main()
