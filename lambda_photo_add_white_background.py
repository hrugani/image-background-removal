import cv2
import os
import boto3


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


def generate_white_background():
    """ """
    try:
        img = cv2.imread(INPUT_FILE_NAME, cv2.IMREAD_UNCHANGED)
        # make mask of where the transparent bits are
        trans_mask = img[:, :, 3] == 0
        # replace areas of transparency with white and not transparent
        img[trans_mask] = [255, 255, 255, 255]
        # new image without alpha channel...
        new_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.basename(OUTPUT_FILE_NAME), new_img)
    except Exception as e:
        raise e


def clear_local_area():
    try:
        os.remove(INPUT_FILE_NAME)
        os.remove(OUTPUT_FILE_NAME)
    except Exception as e:
        raise e


def create_output_file_name(input_file_name):
    return input_file_name.replace("-no-", "-white-")


INPUT_BUCKET_NAME = 'dev-sptrans-photos-no-background'       # must be got from "aws s3 create object event"
INPUT_FILE_NAME = "dne-sample-no-background.png"             # must be got from "aws s3 create object event"
OUTPUT_BUCKET_NAME = 'dev-sptrans-photos-white-background'   # constant - project decision
OUTPUT_FILE_NAME = create_output_file_name(INPUT_FILE_NAME)  # based in INPUT_FILE_NAME


def main():
    download_input_file()
    print("[OK] image:", INPUT_FILE_NAME, "downloaded")
    generate_white_background()
    print("[OK] image:", OUTPUT_FILE_NAME, "generated")
    upload_output_file()
    print("[OK] image:", OUTPUT_FILE_NAME, "Uploaded")
    clear_local_area()
    print("[OK] tmp files:", INPUT_FILE_NAME, OUTPUT_FILE_NAME, "deleted")


if __name__ == '__main__':
    main()



# def download_from_bucket_no_background():
#     pass

# def upload_to_bucket_white_background(white_background_file_name):
#     pass

# file_path = sys.argv[1]
# #load image with alpha channel.  use IMREAD_UNCHANGED to ensure loading of alpha channel
# image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
# #make mask of where the transparent bits are
# trans_mask = image[:,:,3] == 0
# #replace areas of transparency with white and not transparent
# image[trans_mask] = [255, 255, 255, 255]
# #new image without alpha channel...
# new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
# cv2.imwrite(os.path.basename(file_path) + "-while.png", new_img)
