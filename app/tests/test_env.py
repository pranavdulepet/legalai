import os
from dotenv import load_dotenv
load_dotenv('.config')

print(os.getenv('CLOUD_STORAGE_PROVIDER'))
print(os.getenv("AWS_KEY_ID"))
print(os.getenv("AWS_KEY_PASSWORD"))
print(os.getenv("AWS_REGION"))
print(os.getenv("AWS_CONTAINER"))
print(os.getenv("ELASTIC_HOST"))
print(os.getenv("ELASTIC_PORT"))
print(os.getenv("ELASTIC_PORT_1"))
print(os.getenv("load_clf"))
print(os.getenv("load_sim"))
print(os.getenv("sim_type"))
print(os.getenv("load_ner"))
print(os.getenv("load_kp"))
print(os.getenv("load_summ"))
print(os.getenv("load_search"))
