-- train.pig
%default RANDOMSEED '1000'
%default PARTITIONS '3'
%default TRAINFILE  'rcv1_train.multiclass.gz'
%default OUTDIR     'model/'
%default FEATUREBIT '20'
%default FEATURECONVERT 'PARSING'
-- %default FEATURECONVERT 'HASHING'

register ../../../target/pig_udfs-0.0.1.jar;
define Random myudfs.MyRandom('$RANDOMSEED');
training = load '$TRAINFILE' using myudfs.SVMLightLoader() as (label: chararray, features: map[]);
training = foreach training generate label, features, Random() as random;
training = order training by random parallel $PARTITIONS;
training = foreach training generate label, features;
store training into '$OUTDIR' using myudfs.FeaturesMulticlassPerceptronBuilder('$FEATUREBIT', '$FEATURECONVERT');
-- store training into 'model2/' using myudfs.FeaturesPerceptronBuilder('model/');
