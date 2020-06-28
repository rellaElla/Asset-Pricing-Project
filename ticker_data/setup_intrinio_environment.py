import intrinio_sdk
from intrinio_sdk.rest import ApiException
import yaml
import os

here = os.path.dirname(os.path.abspath(__file__))

config_file_name = os.path.join(here, 'setup_intrinio_environment.yaml')



def get_connection():
    with open(config_file_name, 'r') as f:
        vals = yaml.safe_load(f)

    if not ('PRODUCTION_API_KEY' in vals.keys() and
            'SANDBOX_API_KEY' in vals.keys() and
            'USE_PRODUCTION' in vals.keys()):
        raise Exception('Bad config file: ' + config_file_name)

    if vals['USE_PRODUCTION'] is True:
        api_key = vals['PRODUCTION_API_KEY']
        print("Using Production API")
    else:
        api_key = vals['SANDBOX_API_KEY']
        print("Using Sandbox API")

    intrinio_sdk.ApiClient().configuration.api_key['api_key'] = api_key
    return intrinio_sdk


def using_production():
    with open(config_file_name, 'r') as f:
        vals = yaml.safe_load(f)

    if not ('PRODUCTION_API_KEY' in vals.keys() and
            'SANDBOX_API_KEY' in vals.keys() and
            'USE_PRODUCTION' in vals.keys()):
        raise Exception('Bad config file: ' + config_file_name)

    if vals['USE_PRODUCTION'] is True:
        return True
    else:
        return False