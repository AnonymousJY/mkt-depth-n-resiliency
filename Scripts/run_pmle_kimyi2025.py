#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import psycopg2
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

from concurrent.futures import ProcessPoolExecutor
from typing import List
from itertools import product
from Scripts.load_portfolio import get_idiosyncratic_ids
from Library.RiskEngineKimYi2025 import pmle_kimyirisk_systematic, pmle_kimyirisk_idiosyncratic


def pmle_kimyirisk_systematic_helper(args) -> List[tuple]:

    valuation_dt, return_vector, delta_t, seed_number, n_mc_paths, systematic_id, idiosyncratic_id = args

    results = pmle_kimyirisk_systematic(
        sys_returns=return_vector,
        delta_t=delta_t,
        seed_number=seed_number,
        n_mc_paths=n_mc_paths
    )

    results_list = []
    for k, v in results.items():
        stats_name1, stats_name2, stats_name3 = v._asdict().keys()
        stats1, stats2, stats3 = v._asdict().values()
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name1, float(stats1)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name2, float(stats2)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name3, float(stats3)))

    return results_list


def pmle_kimyirisk_idiosyncratic_helper(args) -> List[tuple]:

    valuation_dt, params_sys, return_vector, delta_t, seed_number, n_mc_paths, systematic_id, idiosyncratic_id = args

    results = pmle_kimyirisk_idiosyncratic(
        params_sys=params_sys,
        idi_returns=return_vector,
        delta_t=delta_t,
        seed_number=seed_number,
        n_mc_paths=n_mc_paths
    )

    results_list = []
    for k, v in results.items():
        stats_name1, stats_name2, stats_name3 = v._asdict().keys()
        stats1, stats2, stats3 = v._asdict().values()
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name1, float(stats1)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name2, float(stats2)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name3, float(stats3)))

    return results_list


if __name__=='__main__':

    beg_time = time.perf_counter()

    valuation_beg_dt = '20250331'
    valuation_end_dt = '20250417'
    date_format = '%Y%m%d'
    valuation_window = pd.bdate_range(pd.to_datetime(arg=valuation_beg_dt, format=date_format),
                                      pd.to_datetime(arg=valuation_end_dt, format=date_format))
    valuation_window_str = [dt.strftime(date_format) for dt in valuation_window]

    lookback_period = 252
    base_days = 252
    delta_t = np.array(1 / base_days)
    seed_number = np.uint64(20240114)
    n_mc_paths = int(10_000)

    systematic_id = '^SPX'
    id_dict = {'systematic_id': systematic_id, 'idiosyncratic_ids': get_idiosyncratic_ids()}

    systematic_price_ts = fdr.DataReader(id_dict['systematic_id'])['Adj Close']
    systematic_price_ts.name = id_dict['systematic_id']
    systematic_price_ts.index.name = 'sVALUATION_DATE'

    price_ts = []
    for symbol in id_dict['idiosyncratic_ids']:
        tmp_df = fdr.DataReader(symbol)['Adj Close']
        tmp_df.name = symbol
        tmp_df.index.name = 'sVALUATION_DATE'
        price_ts.append(tmp_df)

    price_ts = pd.concat(price_ts, axis=1)
    price_ts = pd.concat([systematic_price_ts, price_ts], axis=1).ffill().bfill()
    price_ts.index = pd.to_datetime(price_ts.index)
    return_ts = price_ts.pct_change().dropna()

    conn = None
    sys_df = None
    idi_df = None
    set_to_valuate_final_systematic = []
    set_to_valuate_final_idiosyncratic = []

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

        sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sSYSTEMATIC_ID='{systematic_id}' AND sSYSTEMATIC_ID=sIDIOSYNCRATIC_ID;"

        sys_df = pd.read_sql(sql_str, conn)
        sys_df = sys_df.set_index(['svaluation_date', 'sidiosyncratic_id', 'ssystematic_id'])

        array_of_idi_ids = None

        if len(id_dict['idiosyncratic_ids']) > 1:
            array_of_idi_ids = tuple(id_dict['idiosyncratic_ids'])
            sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sIDIOSYNCRATIC_ID IN '{array_of_idi_ids}';"
        else:
            array_of_idi_ids = id_dict['idiosyncratic_ids'][0]
            sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sIDIOSYNCRATIC_ID = '{array_of_idi_ids}';"

        idi_df = pd.read_sql(sql_str, conn)
        idi_df = idi_df.set_index(['svaluation_date', 'sidiosyncratic_id', 'ssystematic_id'])

        print("Successfully fetched data")

        set_to_valuate_total_systematic = list(
            product(valuation_window_str, [id_dict['systematic_id']], [id_dict['systematic_id']]))
        for tuple_to_valuate in set_to_valuate_total_systematic:
            if not tuple_to_valuate in sys_df.index:
                set_to_valuate_final_systematic.append(tuple_to_valuate)

        set_to_valuate_total_idiosyncratic = list(
            product(valuation_window_str, id_dict['idiosyncratic_ids'], [id_dict['systematic_id']]))
        for tuple_to_valuate in set_to_valuate_total_idiosyncratic:
            if not tuple_to_valuate in idi_df.index:
                set_to_valuate_final_idiosyncratic.append(tuple_to_valuate)

    except psycopg2.OperationalError as e:
        print(f"Database not connected successfully: {e}")

    finally:
        conn.close()
        print("Database connection closed")

    # P-MLE Systematic
    if len(set_to_valuate_final_systematic) > 0:
        systematic_arg_list = []
        for dt, idi_id, sys_id in set_to_valuate_final_systematic:
            return_vector = return_ts.loc[return_ts.index <= dt, idi_id].iloc[-lookback_period:].to_numpy()
            systematic_arg_list.append((dt, return_vector, delta_t, seed_number, n_mc_paths, sys_id, idi_id))

        insert_query = ("INSERT INTO pmle_params_kim_yi ("
                        "sVALUATION_DATE, sIDIOSYNCRATIC_ID, sSYSTEMATIC_ID, sPARAMETER, sVALUE_STATISTICS_DESC, sVALUE_STATISTICS"
                        ") VALUES (%s, %s, %s, %s, %s, %s)")

        with ProcessPoolExecutor() as executor:

            results = executor.map(pmle_kimyirisk_systematic_helper, systematic_arg_list)

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

                for result in results:
                    cur.executemany(insert_query, result)
                    num_rows_committed = cur.rowcount
                    print(f"Number of rows affected by the operation: {num_rows_committed}")
                    conn.commit()
                    print("Transaction committed successfully.")

            except psycopg2.OperationalError as e:
                print(f"Database not connected successfully: {e}")

            finally:
                conn.close()
                print("Database connection closed")

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

        sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sSYSTEMATIC_ID='{systematic_id}' AND sSYSTEMATIC_ID=sIDIOSYNCRATIC_ID;"

        sys_df = pd.read_sql(sql_str, conn)
        sys_df = sys_df.set_index(['svaluation_date', 'sidiosyncratic_id', 'ssystematic_id'])

        print("Successfully fetched data")

    except psycopg2.OperationalError as e:
        print(f"Database not connected successfully: {e}")

    finally:
        conn.close()
        print("Database connection closed")

    # P-MLE Idiosyncratic
    idiosyncratic_arg_list = []
    for dt, idi_id, sys_id in set_to_valuate_final_idiosyncratic:
        mask = (sys_df.index.get_level_values('svaluation_date') == dt)
        mask &= (sys_df.index.get_level_values('ssystematic_id') == sys_id)
        mask &= (sys_df.svalue_statistics_desc == 'dMEAN')
        sparameter, svalue = sys_df.loc[mask, ['sparameter', 'svalue_statistics']].to_numpy().T
        params_sys = {k: v for k, v in zip(sparameter, svalue)}
        return_vector = return_ts.loc[return_ts.index <= dt, idi_id].iloc[-lookback_period:].to_numpy()
        idiosyncratic_arg_list.append(
            (dt, params_sys, return_vector, delta_t, seed_number, n_mc_paths, sys_id, idi_id))

    insert_query = ("INSERT INTO pmle_params_kim_yi ("
                    "sVALUATION_DATE, sIDIOSYNCRATIC_ID, sSYSTEMATIC_ID, sPARAMETER, sVALUE_STATISTICS_DESC, sVALUE_STATISTICS"
                    ") VALUES (%s, %s, %s, %s, %s, %s)")

    with ProcessPoolExecutor() as executor:

        results = executor.map(pmle_kimyirisk_idiosyncratic_helper, idiosyncratic_arg_list)

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

            for result in results:
                cur.executemany(insert_query, result)
                num_rows_committed = cur.rowcount
                print(f"Number of rows affected by the operation: {num_rows_committed}")
                conn.commit()
                print("Transaction committed successfully.")

        except psycopg2.OperationalError as e:
            print(f"Database not connected successfully: {e}")

        finally:
            conn.close()
            print("Database connection closed")

    end_time = time.perf_counter()

    elasped_time = end_time - beg_time
    print(f"Time taken: {elasped_time:.6f} seconds")
