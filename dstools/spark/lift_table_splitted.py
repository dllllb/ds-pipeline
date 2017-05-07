import argparse
import os
import sys
import time

from pyhocon import ConfigFactory

start = time.time()

print('{tm} ------------------- {nm} started'.format(
    tm=time.strftime("%Y-%m-%d %H:%M:%S"),
    nm=os.path.basename(__file__)
))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import core as spark_utils

from metrics import lift_splitted

parser = argparse.ArgumentParser()
parser.add_argument('--conf', required=True)
args, overrides = parser.parse_known_args()

file_conf = ConfigFactory.parse_file(args.conf, resolve=False)
overrides = ','.join(overrides)
over_conf = ConfigFactory.parse_string(overrides)
conf = over_conf.with_fallback(file_conf)

sqc = spark_utils.init_session(conf['spark'], app=os.path.basename(args.conf))

lift_cov = lift_splitted(
    sqc,
    query=conf['source.query'],
    target=conf['columns.target'],
    proba=conf['columns.proba'],
    split_by=conf['columns.split-by'],
    cost=conf.get('columns.cost', None),
    n_buckets=int(conf['n_buckets'])
)

lift_cov.to_csv(conf['report-path'], sep='\t')

print('execution time: {} sec'.format(time.time() - start))
