import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

from pydantic import BaseModel
from box import Box
from box.exceptions import BoxKeyError
from datetime import datetime
from fastapi.responses import FileResponse


class FileProp(BaseModel):
  key: str
  last_modified: datetime
  size: str

class FileVersion(BaseModel):
      # print(v.version_id, v.is_latest, v.key, v.last_modified, v.size)
  version_id: str
  is_latest: bool
  key: str
  last_modified: datetime
  size: str


# TODO: merge with class CloudStorage
class FileCloud:

  def __init__(self) -> None:
    load_dotenv('.config')
    AWS_REGION = os.getenv('AWS_REGION')
    ssm = boto3.client('ssm', region_name=AWS_REGION)
    AWS_S3_BUCKET = ssm.get_parameter(Name="/docmansys/prod/docs")
    self.AWS_S3_BUCKET = AWS_S3_BUCKET['Parameter']["Value"]
    self.s3_client = boto3.client('s3')
    self.s3_resource = boto3.resource('s3')
    self.bucket = self.s3_resource.Bucket(name=self.AWS_S3_BUCKET)

  # def upload_file(self, fileobj, account, file_name):
  #   # to improve AWS performance need to randomize file names
  #   # TODO: return the random filename to be saved to database.
  #   # TODO: add file versioning.
  #   # file_name = '_'.join([str(uuid.uuid4().hex[:6]), file_name])
  #   object_name = "/".join(('user_docs', account, file_name))
  #   try:
  #     response = self.s3_client.upload_fileobj(fileobj,
  #                                             self.AWS_S3_BUCKET,
  #                                             object_name)
  #     print(response) # TODO: turn off in production
  #   except ClientError as e:
  #     print(e)
  #     return False
  #   return file_name

  def upload_file(self, fileobj, account, prefix, file_name):
    if prefix:
      object_name = "/".join(('user_docs', account, prefix, file_name))
    else:
      object_name = "/".join(('user_docs', account, file_name))

    obj = self.bucket.Object(object_name)
    try:
      obj.upload_fileobj(fileobj)
    except ClientError as e:
      print(e)
      return False
    return True

  def list_files(self, account, prefix=None, max_items=None, prev_token=None):
    if prefix:
      object_name = "/".join(('user_docs', account, prefix))
    else:
      object_name = "/".join(('user_docs', account))

    kwargs = {
              'Bucket': self.AWS_S3_BUCKET,
              'Prefix': object_name,
            }
    if max_items:
      kwargs['MaxKeys'] = 2
    if prev_token:
      kwargs['ContinuationToken'] = prev_token

    obj_list = self.s3_client.list_objects_v2(**kwargs)
    print(obj_list)
    obj_list = Box(obj_list)

    try:
      next_token = obj_list.NextContinuationToken
    except BoxKeyError as e:
      print(f"No such key: NextContinuationToken")
      next_token = ""

    file_list = []
    try:
      for obj in obj_list.Contents:
        # b_obj = Box(obj)
        print(obj.Key, obj.LastModified, obj.Size)
        file_list.append(FileProp(key = obj.Key,
                                  last_modified = obj.LastModified,
                                  size = obj.Size))
    except BoxKeyError as e:
      print(f"No files found")
      print(e)

    # return file_list
    return {"file_list": file_list, "next_token": next_token}

  def list_files_page(self, account, prefix=None, prev_token=""):
    if prefix:
      object_name = "/".join(('user_docs', account, prefix))
    else:
      object_name = "/".join(('user_docs', account))

    if prev_token:
      pagination_config = {'MaxItems': 2, 'StartingToken': prev_token, 'PageSize': 2}
    else:
      pagination_config = {'MaxItems': 2, 'PageSize': 2}

    paginator = self.s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=self.AWS_S3_BUCKET,
                                        Prefix=object_name,
                                        PaginationConfig=pagination_config)

    file_list = []
    next_token = ""
    print(type(page_iterator))
    print(page_iterator)
    for obj_list in page_iterator:
      obj_list = Box(obj_list)
      print(obj_list)
      for obj in obj_list.Contents:
        # b_obj = Box(obj)
        print(obj.Key, obj.LastModified, obj.Size)
        file_list.append(FileProp(key = obj.Key,
                                  last_modified = obj.LastModified,
                                  size = obj.Size))
      try:
        next_token = obj_list.NextContinuationToken
      except BoxKeyError as e:
        print(f"No such key: NextContinuationToken")

    return {"file_list": file_list, "next_token": next_token}

  def list_file_versions(self, account, file_path=""):
    object_name = "/".join(('user_docs', account, file_path))
    versions = self.bucket.object_versions.filter(Prefix=object_name)
    file_versions = []
    for v in versions:
      print(v.version_id, v.is_latest, v.key, v.last_modified, v.size)
      file_versions.append(
        FileVersion(version_id = v.version_id,
                    is_latest = v.is_latest,
                    key = v.key,
                    last_modified = v.last_modified,
                    size = v.size))
    return file_versions

  def get_file(self, account, file_path="", version_id=""):
    object_name = "/".join(('user_docs', account, file_path))
    obj = self.bucket.Object(object_name)
    print(obj)
    try:
      if version_id:
        print(f"version_id: {version_id}")
        response = obj.get(VersionId = version_id)
      else:
        response = obj.get()
    except ClientError as e:
      if e.response['Error']['Code'] == 'NoSuchKey':
        print("No such file exists")
        return "No such file exists, or not enough priviledges"
      else:
        print(e.response)
        return "Error reading file"
    # print(response)
    return response['Body'].read()

  def get_all_files(self, account, prefix=None, max_items=None, prev_token=None):
    if prefix:
      object_name = "/".join(('user_docs', account, prefix))
    else:
      object_name = "/".join(('user_docs', account))

    kwargs = {
              'Bucket': self.AWS_S3_BUCKET,
              'Prefix': object_name,
            }
    if max_items:
      kwargs['MaxKeys'] = max_items
    if prev_token:
      kwargs['ContinuationToken'] = prev_token

    obj_list = self.s3_client.list_objects_v2(**kwargs)

    try:
      next_token = obj_list["NextContinuationToken"]
    except BoxKeyError as e:
      print(f"No such key: NextContinuationToken")
      next_token = ""

    file_list = []
    obj_list = obj_list["Contents"]
    for obj in obj_list:
      object_name = obj["Key"]
      obj = self.bucket.Object(object_name)
      response = obj.get()
      file_list.append(response['Body'].read())

    print(f"Number of files: {len(file_list)}")

    # return file_list
    return {"file_list": file_list, "next_token": next_token}

  def download_file(self, account, file_path="", version_id=""):
    object_name = "/".join(('user_docs', account, file_path))
    obj = self.bucket.Object(object_name)
    print(obj)
    try:
      if version_id:
        print(f"version_id: {version_id}")
        # response = obj.download_fileobj(fp, ExtraArgs={'VersionId': version_id})
        response = obj.download_file('junk.txt', ExtraArgs={'VersionId': version_id})
      else:
        response = obj.download_file('junk.txt')
        # response = obj.download_fileobj(fp)
    except ClientError as e:
      if e.response['Error']['Code'] == 'NoSuchKey':
        print("No such file exists")
        return "No such file exists"
      else:
        return e.response
    print(response)
    # TODO: save this to tmp files and delete later
    return FileResponse(path='junk.txt', headers={"Content-Disposition": "attachment; filename=junk.txt"})

  def delete_file(self, account, file_path="", version_id=""):
    object_name = "/".join(('user_docs', account, file_path))
    obj = self.bucket.Object(object_name)
    try:
      if version_id:
        print(f"version_id: {version_id}")
        # response = obj.get(VersionId = version_id)
        response = obj.delete(VersionId = version_id)
      else:
        # response = obj.get()
        response = obj.delete()
    except ClientError as e:
      return e.response
    return response
