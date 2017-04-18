def pandify(sdf):
    res = sdf

    final_cols = []

    for f in res.schema.fields:
        if '.' in f.name:
            new_name = f.name.replace('.', '__')
            field_name = '`{nm}`'.format(nm=f.name)
            res = res.withColumn(new_name, res[field_name])

        final_cols.append(f.name.replace('.', '__'))

    res = res.select(*final_cols)

    for f in res.schema.fields:
        if f.dataType.typeName() == 'decimal':
            res = res.withColumn(f.name, res[f.name].astype('float'))

    return res


def limit(sdf, n_records):
    res = sdf.rdd.zipWithIndex()\
        .filter(lambda e: e[1] < n_records)\
        .map(lambda e: e[0]).toDF()
    return res


def score(sc, sdf, model_path, cols_to_save, target_class=1):
    from sklearn.externals import joblib
    import json

    def block_iterator(iterator, size):
        bucket = list()
        for e in iterator:
            bucket.append(e)
            if len(bucket) >= size:
                yield bucket
                bucket = list()
        if bucket:
            yield bucket

    model = joblib.load(model_path)
    model_bc = sc.broadcast(model)
    col_bc = sc.broadcast(sdf.columns)

    def block_classify(iterator):
        import pandas as pd

        for features in block_iterator(iterator, 10000):
            features_df = pd.DataFrame(list(features), columns=col_bc.value)
            existing_cols_to_save = list(set(cols_to_save).intersection(features_df.columns))
            res_df = features_df[existing_cols_to_save].copy()

            res_df['target_proba'] = model_bc.value.predict_proba(features_df)[:, target_class]

            for e in json.loads(res_df.to_json(orient='records')):
                yield e

    scores = sdf.mapPartitions(block_classify)
    score_df = scores.toDF()

    return score_df


def define_data_frame(conf, sqc):
    storage = conf['storage']

    if storage == 'jdbc':
        sdf = jdbc_load(
            sqc=sqc,
            query=conf['query'],
            conn_params=conf['conn'],
            partition_column=conf.get('partition-column', None),
            num_partitions=conf.get('num-partitions', None))
    elif storage == 'local':
        dataset_format = conf.get('dataset-store-format', 'parquet')
        data_path = conf['query']
        sdf = sqc.read.format(dataset_format).load(data_path, header=True)
    elif storage == 'hdfs':
        dataset_format = conf.get('dataset-store-format', 'parquet')
        data_path = conf['query']
        sdf = sqc.read.format(dataset_format).load(data_path, header=True)
    elif storage == 'single-csv':
        data_path = conf['query']
        header = conf.get('header', 'infer')
        sep = conf.get('sep', '\t')
        decimal = conf.get('decimal', '.')
        import pandas as pd
        pdf = pd.read_csv(data_path, sep=sep, header=header, decimal=decimal, encoding='utf8')
        sdf = sqc.createDataFrame(pdf)
    elif storage == 'hive':
        sdf = sqc.sql(conf['query'])
    else:
        raise ValueError('unknown storage type: {st}'.format(st=storage))

    if 'distribute-by' in conf:
        sdf = sdf.repartition(conf['distribute-by.n-partitions'], conf['distribute-by.key'])

    if 'transform-sql' in conf:
        sdf.registerDataFrameAsTable(sdf, 'dataset_temp')
        sdf = sqc.sql(conf['transform-sql'])

    if 'sample' in conf:
        sdf = sdf.sample(False, fraction=conf.get_float('sample'), seed=4233)

    if 'limit' in conf:
        sdf = sdf.limit(conf.get_int('limit'))

    return sdf


def write(conf, sdf):
    if conf.get_bool('disabled', False):
        return

    storage = conf['storage']

    if 'distribute-by' in conf:
        sdf = sdf.repartition(conf['distribute-by.n-partitions'], conf['distribute-by.key'])

    if 'n-partitions' in conf:
        sdf = sdf.repartition(conf['n-partitions'])

    if storage == 'local':
        target_dir = conf['query']
        import os
        if not os.path.exists(os.path.dirname(target_dir)):
            os.makedirs(os.path.dirname(target_dir))

        write_format = conf.get('dataset-store-format', 'parquet')
        write_mode = conf.get('write-mode', 'overwrite')
        sdf.write.mode(write_mode).format(write_format).save(target_dir)
    elif storage == 'hdfs':
        target_dir = conf['query']
        write_format = conf.get('dataset-store-format', 'parquet')
        write_mode = conf.get('write-mode', 'overwrite')

        w = sdf.write.mode(write_mode).format(write_format)

        partition_by = conf.get('partition-by', None)
        w.save(target_dir, partitionBy=partition_by)
    elif storage == 'jdbc':
        result_table = conf['query']

        write_mode = conf.get('write-mode', 'append')

        sdf.repartition(1).write.mode(write_mode).jdbc(
            url=conf['conn']['url'],
            properties=conf['conn'],
            table=result_table)
    elif storage == 'hive':
        write_mode = conf.get('write-mode', 'append')
        table = conf['query']

        db, tname = table.split('.')
        if tname in sdf.sql_ctx.tableNames(db):
            cols = sdf.sql_ctx.sql('show columns in {}'.format(table)).toPandas().result.tolist()
            column_order = [c.strip() for c in cols]
        else:
            column_order = sdf.columns

        w = sdf.select(*column_order).write.mode(write_mode)

        write_format = conf.get('dataset-store-format', None)
        if write_format is not None:
            w = w.format(write_format)

        partition_by = conf.get('partition-by', None)
        w.saveAsTable(table, partitionBy=partition_by)
    elif storage == 'single-csv':
        data_path = conf['query']
        header = conf.get_bool('header', True)
        sep = conf.get('sep', '\t')
        decimal = conf.get('decimal', '.')
        pdf = sdf.toPandas()
        pdf.to_csv(data_path, sep=sep, header=header, decimal=decimal, encoding='utf8')
    else:
        raise ValueError('unknown storage type: {st}'.format(st=storage))


def prop_list(tree, prefix=list()):
    res = dict()
    for k, v in tree.items():
        path = prefix+[k]
        if isinstance(v, dict):
            res.update(prop_list(v, path))
        else:
            res['.'.join(path)] = v
    return res


def init_spark(config, app=None, use_session=False):
    import os
    import sys
    from glob import glob

    if 'spark-home' in config:
        os.environ['SPARK_HOME'] = config['spark-home']

    if 'spark-conf-dir' in config:
        os.environ['SPARK_CONF_DIR'] = config['spark-conf-dir']

    if 'pyspark-python' in config:
        # Set python interpreter on both driver and workers
        os.environ['PYSPARK_PYTHON'] = config['pyspark-python']

    if 'yarn-conf-dir' in config:
        # Hadoop YARN configuration
        os.environ['YARN_CONF_DIR'] = config['yarn-conf-dir']

    if 'spark-classpath' in config:
        # can be used to use external folder with Hive configuration
        # e. g. spark-classpath='/etc/hive/conf.cloudera.hive1'
        os.environ['SPARK_CLASSPATH'] = config['spark-classpath']

    submit_args = []

    driver_mem = config.get('spark-prop.spark.driver.memory', None)
    if driver_mem is not None:
        submit_args.extend(["--driver-memory", driver_mem])

    driver_cp = config.get('spark-prop.spark.driver.extraClassPath', None)
    if driver_cp is not None:
        submit_args.extend(["--driver-class-path", driver_cp])

    driver_java_opt = config.get('spark-prop.spark.driver.extraJavaOptions', None)
    if driver_java_opt is not None:
        submit_args.extend(["--driver-java-options", driver_java_opt])

    jars = config.get('jars', None)
    if jars is not None:
        if isinstance(jars, str):
            jars = [jars]
        submit_args.extend(["--jars", ','.join(jars)])

    mode_yarn = config['spark-prop.spark.master'].startswith('yarn')

    if mode_yarn:
        # pyspark .zip distribution flag is set only if spark-submit have master=yarn in command-line arguments
        # see spark.yarn.isPython conf property setting code
        # in org.apache.spark.deploy.SparkSubmit#prepareSubmitEnvironment
        submit_args.extend(['--master', 'yarn'])

    # pyspark .zip distribution flag is set only if spark-submit have pyspark-shell or .py as positional argument
    # see spark.yarn.isPython conf property setting code
    # in org.apache.spark.deploy.SparkSubmit#prepareSubmitEnvironment
    submit_args.append('pyspark-shell')

    os.environ['PYSPARK_SUBMIT_ARGS'] = ' '.join(submit_args)

    spark_home = os.environ['SPARK_HOME']
    spark_python = os.path.join(spark_home, 'python')
    pyspark_libs = glob(os.path.join(spark_python, 'lib', '*.zip'))
    sys.path.extend(pyspark_libs)

    if use_session:
        from pyspark.sql import SparkSession

        builder = SparkSession.builder.appName(app or config['app'])

        if mode_yarn:
            builder = builder.enableHiveSupport()

        for k, v in prop_list(config['spark-prop']).items():
            builder = builder.config(k, v)

        ss = builder.getOrCreate()
        return ss
    else:
        from pyspark import SparkConf, SparkContext
        conf = SparkConf()
        conf.setAppName(app or config['app'])
        props = [(k, str(v)) for k, v in prop_list(config['spark-prop']).items()]
        conf.setAll(props)
        sc = SparkContext(conf=conf)
        return sc


def init_session(config, app=None, return_context=False, overrides=None, use_session=False):
    if isinstance(config, str):
        import os
        from pyhocon import ConfigFactory
        if os.path.exists(config):
            if overrides is not None:
                file_conf = ConfigFactory.parse_file(config, resolve=False)
                over_conf = ConfigFactory.parse_string(overrides)
                conf = over_conf.with_fallback(file_conf)
            else:
                conf = ConfigFactory.parse_file(config)
        else:
            conf = ConfigFactory.parse_string(config)
    else:
        conf = config

    res = init_spark(conf, app, use_session)

    if use_session:
        return res
    else:
        mode_yarn = conf['spark-prop.spark.master'].startswith('yarn')

        if mode_yarn:
            from pyspark.sql import HiveContext
            sqc = HiveContext(res)

            if 'hive-prop' in conf:
                for k, v in prop_list(conf['hive-prop']).items():
                    sqc.setConf(k, str(v))
        else:
            from pyspark.sql import SQLContext
            sqc = SQLContext(res)

        if return_context:
            return res, sqc
        else:
            return sqc


def jdbc_load(
    sqc,
    query,
    conn_params,
    partition_column=None,
    num_partitions=10,
    fetch_size=10000000
):
    import re
    if re.match('\s*\(.+\)\s+as\s+\w+\s*', query):
        _query = query
    else:
        _query = '({}) as a'.format(query)

    conn_params_base = dict(conn_params)
    if partition_column and num_partitions and num_partitions > 1:
        min_max_query = '''
          (select max({part_col}) as max_part, min({part_col}) as min_part
             from {query}) as g'''.format(part_col=partition_column, query=_query)
        max_min_df = sqc.read.load(dbtable=min_max_query, **conn_params_base)
        tuples = max_min_df.rdd.collect()
        max_part = str(tuples[0].max_part)
        min_part = str(tuples[0].min_part)
        conn_params_base['fetchSize'] = str(fetch_size)
        conn_params_base['partitionColumn'] = partition_column
        conn_params_base['lowerBound'] = min_part
        conn_params_base['upperBound'] = max_part
        conn_params_base['numPartitions'] = str(num_partitions)
    sdf = sqc.read.load(dbtable=_query, **conn_params_base)
    return sdf
