import tarfile
import pathlib
from datetime import datetime as dt
from libcloud.storage.types import Provider
from libcloud.storage.providers import get_driver
import os


class CloudStorage:
    @classmethod
    def get_files(cls) -> None:
        ''' Downloads the model and data from hosted Cloud Service '''
        name = os.getenv('CLOUD_STORAGE_PROVIDER')
        if name == "AWS":
            return cls.get_from_aws()
        else:
            raise Exception(f'Unsupported cloud provider: {name}')

    @classmethod
    def get_from_aws(cls):
        # TODO: change to boto3
        client = get_driver(Provider.S3)
        driver = client(os.getenv("AWS_KEY_ID"),
                        os.getenv("AWS_KEY_PASSWORD"),
                        region=os.getenv("AWS_REGION"))

        container = driver.get_container(container_name=os.getenv("AWS_CONTAINER"))
        return container.list_objects()


def _download():
    ''' Download processed corpus from cloud '''
    objects = CloudStorage.get_files()
    print(objects)

    for obj in objects:
        dest = pathlib.Path.cwd() / obj.name
        print(f'downloading {obj.name} to {dest}')
        stime = dt.now()
        # return false if download fails.
        obj.download(destination_path=dest, overwrite_existing=True)
        print(f'time for downloading {dt.now() - stime}')
        with tarfile.open(dest) as tfile:
            print(f'extracting {obj.name}')
            stime = dt.now()
            tfile.extractall('./')
            print(f'time for extracting {dt.now() - stime}')
        os.remove(dest) # TODO: test this
        print('\n...................\n')
    return True


if __name__ == "__main__":
    _download()
