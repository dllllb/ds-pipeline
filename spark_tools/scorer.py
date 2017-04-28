import argparse
import os
import shutil
import sys
import time

from pyhocon import ConfigFactory

start = time.time()

print('{tm} ------------------- {nm} started'.format(
    tm=time.strftime("%Y-%m-%d %H:%M:%S"),
    nm=os.path.basename(__file__)
))

module_path = os.path.realpath(__file__)
sys.path.append(module_path)

import core as spark_utils

parser = argparse.ArgumentParser()
parser.add_argument('--conf', required=True)
args, overrides = parser.parse_known_args()

file_conf = ConfigFactory.parse_file(args.conf, resolve=False)
overrides = ','.join(overrides)
over_conf = ConfigFactory.parse_string(overrides)
conf = over_conf.with_fallback(file_conf)

sc, sqc = spark_utils.init_session(conf['spark'], app=os.path.basename(args.conf), return_context=True)

module_dir = os.path.dirname(os.path.dirname(module_path))
os.makedirs(os.path.expanduser('~/.temp'))
zip_path = '~/.temp/ds-tools.zip'
shutil.make_archive(zip_path, 'zip', module_dir)
sc.addPyFile(zip_path)

pipeline_file = conf.get('pipeline-file', None)

if pipeline_file is not None:
    pipeline_full_path = os.path.join(
        os.path.dirname(os.path.realpath(args.conf)), pipeline_file)
    sc.addPyFile(pipeline_full_path)

print('{} loading data...'.format(time.strftime("%Y-%m-%d %H:%M:%S")))

sdf = spark_utils.define_data_frame(conf['source'], sqc)
sdf = sdf.filter('uid is not null')
sdf = sdf.withColumn('uid', sdf.uid.astype('string'))
sdf = spark_utils.pandify(sdf)

score_df = spark_utils.score(
    sc=sc,
    sdf=sdf,
    model_path=os.path.expanduser(conf['model-path']),
    cols_to_save=['uid', 'true_target', 'business_dt']
).cache()

author_name = os.environ.get('USER', '!unknown')
model_name = os.path.basename(conf['model-path'])
current_dt = time.strftime("%Y-%m-%dT%H-%M")

score_df = score_df.selectExpr(
    "'{}' as author".format(author_name),
    "'{}' as model_name".format(model_name),
    "'{}' as current_dt".format(current_dt),
    '*'
)

print('scores generated: {}'.format(score_df.count()))

print('{} saving scores ...'.format(time.strftime("%Y-%m-%d %H:%M:%S")))

spark_utils.write(conf['target'], score_df)

print('execution time: {} sec'.format(time.time() - start))
