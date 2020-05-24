from unittest import TestCase
import unittest
import lambda_photo_add_white_background as t


class Test(TestCase):

    def test_get_s3_client(self):
        s3 = t.get_s3_client()
        resp = s3.__dict__.__len__()
        self.assertEqual(resp, 10)
