#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from collections import namedtuple


Portfolio = namedtuple(
    typename='Portfolio',
    field_names=[
        'sVALUATION_DATE',
        'sIDIOSYNCRATIC_ID',
        'sSTRATEGY',
        'sTYPOLOGY',
        'sPAYOFF_TYPE',
        'sEXPIRY_DATE',
        'dSTRIKE_PRICE',
        'dPOSITION',
        'sFIXING_FREQUENCY',
        'sFIXING_DATE_BEG',
        'sFIXING_DATE_END',
        'dBASE_DAYS'
    ]
)


def get_idiosyncratic_ids() -> List[str]:
    names = ["COIN"]
    return names


if __name__=="__main__":
    import psycopg2
    import pandas as pd

    valuation_beg_dt = '20250331'
    valuation_end_dt = '20250416'
    expiry_date = '20250417'

    collar_strike_put  = 172.23
    collar_strike_call = 198.06
    vanilla_strike_put  = 180.84
    vanilla_strike_call = 206.67

    collar_position_put  = -100.
    collar_position_call = 100.
    vanilla_position_put  = 85.
    vanilla_position_call = -120.

    dt_format = '%Y%m%d'
    base_days = 252
    fixing_frequency = 'B'
    fixing_date_beg = '20250331'
    fixing_date_end = '20250417'

    valuation_date_window = pd.bdate_range(
        start=pd.to_datetime(arg=valuation_beg_dt, format=dt_format),
        end=pd.to_datetime(arg=valuation_end_dt, format=dt_format)
    )

    # print(valuation_date_window)

    conn = None

    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        print("Database connected successfully")

        portfolio_to_insert = []

        for val_dt in valuation_date_window:
            for idiosyncratic_id in get_idiosyncratic_ids():
                val_dt = val_dt.strftime(dt_format)
                portfolio1 = Portfolio(
                    sVALUATION_DATE=val_dt,
                    sIDIOSYNCRATIC_ID=idiosyncratic_id,
                    sTYPOLOGY='collar',
                    sSTRATEGY='asian discrete',
                    sPAYOFF_TYPE='put',
                    sEXPIRY_DATE=expiry_date,
                    dSTRIKE_PRICE=collar_strike_put,
                    dPOSITION=collar_position_put,
                    sFIXING_FREQUENCY=fixing_frequency,
                    sFIXING_DATE_BEG=fixing_date_beg,
                    sFIXING_DATE_END=fixing_date_end,
                    dBASE_DAYS=base_days
                )
                portfolio2 = Portfolio(
                    sVALUATION_DATE=val_dt,
                    sIDIOSYNCRATIC_ID=idiosyncratic_id,
                    sTYPOLOGY='collar',
                    sSTRATEGY='asian discrete',
                    sPAYOFF_TYPE='call',
                    sEXPIRY_DATE=expiry_date,
                    dSTRIKE_PRICE=collar_strike_call,
                    dPOSITION=collar_position_call,
                    sFIXING_FREQUENCY=fixing_frequency,
                    sFIXING_DATE_BEG=fixing_date_beg,
                    sFIXING_DATE_END=fixing_date_end,
                    dBASE_DAYS=base_days
                )
                portfolio3 = Portfolio(
                    sVALUATION_DATE=val_dt,
                    sIDIOSYNCRATIC_ID=idiosyncratic_id,
                    sTYPOLOGY='vanilla',
                    sSTRATEGY='vanilla',
                    sPAYOFF_TYPE='put',
                    sEXPIRY_DATE=expiry_date,
                    dSTRIKE_PRICE=vanilla_strike_put,
                    dPOSITION=vanilla_position_put,
                    sFIXING_FREQUENCY=None,
                    sFIXING_DATE_BEG=None,
                    sFIXING_DATE_END=None,
                    dBASE_DAYS=base_days
                )
                portfolio4 = Portfolio(
                    sVALUATION_DATE=val_dt,
                    sIDIOSYNCRATIC_ID=idiosyncratic_id,
                    sTYPOLOGY='vanilla',
                    sSTRATEGY='vanilla',
                    sPAYOFF_TYPE='call',
                    sEXPIRY_DATE=expiry_date,
                    dSTRIKE_PRICE=vanilla_strike_call,
                    dPOSITION=vanilla_position_call,
                    sFIXING_FREQUENCY=None,
                    sFIXING_DATE_BEG=None,
                    sFIXING_DATE_END=None,
                    dBASE_DAYS=base_days
                )
                portfolio_to_insert.append(portfolio1)
                portfolio_to_insert.append(portfolio2)
                portfolio_to_insert.append(portfolio3)
                portfolio_to_insert.append(portfolio4)

        insert_query = ("INSERT INTO portfolio ("
                        "sVALUATION_DATE, sIDIOSYNCRATIC_ID, sTYPOLOGY, sSTRATEGY, sPAYOFF_TYPE, "
                        "sEXPIRY_DATE, dSTRIKE_PRICE, dPOSITION, sFIXING_FREQUENCY, sFIXING_DATE_BEG, "
                        "sFIXING_DATE_END, dBASE_DAYS"
                        ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

        cur.executemany(insert_query, portfolio_to_insert)

        conn.commit()

        print(f"{cur.rowcount} rows inserted successfully.")

    except psycopg2.OperationalError as e:
        print(f"Database not connected successfully: {e}")

    finally:
        conn.close()