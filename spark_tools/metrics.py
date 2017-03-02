def lift_splitted(sqc,
                  query,
                  target='true_target',
                  proba='target_proba',
                  split_by={'model_name, business_dt'},
                  cost=None,
                  n_buckets=100):
    """
    Calculate lift function over splits. Requires sqlContext
    Input:
        sqc - sqlContext of spark session
        query - query Dataframe with true_target, target_proba and split columns
        split_by - list of columns to calculate lift independently
        target - binary actual values to predict
        cost - optional, charges
        proba - probabilities of target = 1
        n_buckets - number of buckets to bin lift function
    Output:
        pdf - pandas DataFrame with cumulative lift and coverage
    """
    import pandas as pd
    import pyspark.sql.functions as F

    sqc.sql(query).registerTempTable("tmp_lift_splitted")
    # generate ntiles
    if cost is not None:
        sql = """select {sb}, CAST({target} AS INT) {target}, {proba}, {cost},
                 NTILE({nb}) OVER (PARTITION BY {sb} ORDER BY {proba} DESC) as tile
                 FROM tmp_lift_splitted t
              """.format(sb=split_by, target=target, proba=proba, cost=cost, nb=n_buckets)
    else:
        sql = """select {sb}, CAST({target} AS INT) {target}, {proba},
                 NTILE({nb}) OVER (PARTITION BY {sb} ORDER BY {proba} DESC) as tile
                 FROM tmp_lift_splitted t
              """.format(sb=split_by, target=target, proba=proba, cost=cost, nb=n_buckets)

    sdf = sqc.sql(sql)
    if cost is not None:
        pdf = sdf.groupby(split_by.union({'tile'})).agg(
            F.sum(F.col(target)).alias("target_sum"),
            F.count(F.col(target)).alias("target_cnt"),
            F.min(F.col(proba)).alias("target_proba_min"),
            F.max(F.col(proba)).alias("target_proba_max"),
            F.sum(F.col(cost)).alias("cost_sum"),
            F.sum(F.col(cost).isNotNull().cast('integer')).alias("cost_cnt")
        ).toPandas()
        pdf['cost_sum'] = pdf['cost_sum'].astype(float)
    else:
        pdf = sdf.groupby(split_by.union({'tile'})).agg(
            F.sum(F.col(target)).alias("target_sum"),
            F.count(F.col(target)).alias("target_cnt"),
            F.min(F.col(proba)).alias("target_proba_min"),
            F.max(F.col(proba)).alias("target_proba_max")
        ).toPandas()

    if 'business_dt' in split_by:
        pdf['business_dt'] = pd.to_datetime(pdf['business_dt'], errors='coerce')

    pdf = pdf.sort_values('tile')
    grouped = pdf.groupby(split_by, as_index=False)
    pdf['target_cum_sum'] = grouped.target_sum.cumsum()
    pdf['target_cum_cnt'] = grouped.target_cnt.cumsum()

    if cost is not None:
        pdf['charge_cum_sum'] = grouped.cost_sum.cumsum()
        pdf['charge_cum_cnt'] = grouped.cost_cnt.cumsum()

    pdf = pdf.sort_values(split_by).set_index(split_by)
    pdf['response_cum'] = pdf.target_cum_sum / pdf.target_cum_cnt
    pdf['target_sum_base'] = pdf.loc[pdf.tile == n_buckets, 'target_cum_sum']
    pdf['target_cnt_base'] = pdf.loc[pdf.tile == n_buckets, 'target_cum_cnt']
    pdf['lift'] = pdf['response_cum'] / (pdf['target_sum_base'] / pdf['target_cnt_base'])
    pdf['coverage'] = pdf['target_cum_sum'] / pdf['target_sum_base']

    if cost is not None:
        pdf['charge_average'] = pdf.cost_sum / pdf.cost_cnt
        pdf['charge_gain'] = pdf.charge_cum_sum / pdf.loc[pdf.tile == n_buckets, 'charge_cum_sum'] \
            .loc[pdf.index]

    return pdf
