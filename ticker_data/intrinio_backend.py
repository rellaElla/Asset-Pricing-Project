import intrinio_sdk
from intrinio_sdk.rest import ApiException
import setup_intrinio_environment

import pandas as pd
import math

import time

intrinio_sdk = setup_intrinio_environment.get_connection()
security_api = intrinio_sdk.SecurityApi()
company_api = intrinio_sdk.CompanyApi()
BULK_API_CALL_LIMIT = 3

def fetch_daily_price(identifier, start_date='', end_date='', next_page='', page_size=None):
    """
    Description: Fetches daily prices of a security, indicated by the indetifier. Can be ticker, FIGI, etc.
    Returns a pandas DataFrame. 
    """

     # https://docs.intrinio.com/documentation/python/get_security_stock_prices_v2
    page_size=page_size

    if not start_date:
        page_size: int = 10000

    elif start_date and not end_date and not page_size:
        page_size = int(math.ceil((pd.to_datetime('today') - pd.to_datetime(start_date)).days/100)) * 100

    elif start_date and end_date and not page_size:
        page_size = int(math.ceil((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days/100)) * 100

    try:
        if page_size<=100:
            api_response = security_api.get_security_stock_prices(
                                                        identifier,
                                                        start_date=start_date,
                                                        frequency='daily',
                                                        page_size=page_size,
                                                        next_page=next_page)

        else:
            api_response = security_api.get_security_stock_prices(
                                            identifier,
                                            start_date=start_date,
                                            frequency='daily',
                                            page_size=page_size,
                                            next_page=next_page)
            time.sleep(BULK_API_CALL_LIMIT)
    except ApiException as e:
        print("Exception: SecurityApi->get_security_historical_data: %s\n" % e)
        time.sleep(BULK_API_CALL_LIMIT)
        return None

    return pd.DataFrame(api_response.stock_prices_dict)

def get_all_securities(next_page = '', currency = 'USD', composite_mic = 'USCOMP'):
    intrinio_sdk = setup_intrinio_environment.get_connection()
    security_api = intrinio_sdk.SecurityApi()
    try:
        api_response = security_api.get_all_securities(
            active = True,
            currency = currency,
            composite_mic = composite_mic,
            next_page = next_page,
            page_size = 10000
        )

    except ApiException as e:
        print('Exception: SecurityApi -> get_all_securities: %s\r\n' % e)
        return None

    return pd.DataFrame(api_response.securities_dict)