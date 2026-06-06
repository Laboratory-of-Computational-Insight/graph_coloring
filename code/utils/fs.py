# @title File System infra

import glob
import os
from typing import Optional

import boto3
import botocore

from utils.secrets import WASABI_KEY, WASABI_SECERT, wasabi_url, bucket
from utils.singlton import Singleton

class FS(metaclass=Singleton):
    def __init__(self, read_timeout=300, connect_timeout=300):
        # if url is None or key is None or secret is None or bucket is None:
        #     return

        self.s3 = boto3.resource(
            "s3",
            endpoint_url=wasabi_url,
            aws_access_key_id=WASABI_KEY,
            aws_secret_access_key=WASABI_SECERT,
            config=botocore.client.Config(read_timeout=read_timeout, connect_timeout=connect_timeout),
        )
        self.boto_bucket = self.s3.Bucket(bucket)
        self.bucket = bucket

    def upload_local_directory(self, local_path, remote_path):
        if not os.path.isdir(local_path):
            print(f"{local_path} is not a dir")
        for local_file in glob.glob(local_path + "/**"):
            if not os.path.isfile(local_file):
                self.upload_local_directory(
                    local_file, remote_path + "/" + os.path.basename(local_file)
                )
            else:
                remote_path = os.path.join(
                    remote_path, local_file[1 + len(local_path) :]
                )
                self.upload_file(local_file, remote_path)

    def upload_file(self, local, remote):
        self.boto_bucket.upload_file(local, remote)

    def upload_data(self, data, remote):
        object = self.s3.Object(self.bucket, remote.replace("\\", "/"))
        object.put(Body=data)

    def get_file(self, remote, local=None, download=True, override=False):
        old_sep = os.sep
        os.sep = '/'

        if local is None:
            local = os.path.normpath(os.path.join("tmp", *os.path.split(remote)))

        local_folder = os.path.dirname(local)
        os.makedirs(local_folder, exist_ok=True)

        if download and not override:
            if os.path.exists(local):
                return local

        self.boto_bucket.download_file(remote, local)

        os.sep = old_sep
        return local

    def get_data(self, remote, delimiter=""):
        if not delimiter:
            delimiter=remote
        body = None
        for obj in self.boto_bucket.objects.filter(Prefix=delimiter):
            key = obj.key
            if key == remote:
                body = obj.get()["Body"].read()
        return body

    def list_items(self, prefix:str, validate_suffix:str = "", negative_validate_suffix:Optional[str]=None):
        prefix = prefix.replace("\\", "/")
        return [
            obj.key for obj in self.boto_bucket.objects.filter(Prefix=prefix)
            if obj.key.endswith(validate_suffix) and (negative_validate_suffix is None or not obj.key.endswith(negative_validate_suffix))
        ]

    def file_exists(self, file):
        # file=file.replace("\\", "/")
        dir = os.path.dirname(file)
        for f in self.boto_bucket.objects.filter(Prefix=dir):
            if f.key == file:
                return True
        return False

    def folder_exists(self, file):
        file = file.replace("\\", "/")
        return len(list(self.boto_bucket.objects.filter(Prefix=file))) > 0

    def delete(self, file):
        self.s3.Object(self.bucket, file).delete()

    def move(self, src, dst):
        self.s3.Object(self.bucket, dst).copy_from(CopySource=f"{self.bucket}/{src}")
