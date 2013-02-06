-- train.pig
%default RANDOMSEED '1000'
%default PARTITIONS '3'
%default TRAINFILE  'rcv1_train.binary.gz'
%default OUTDIR     'model/'
%default FEATUREBIT '20'
%default FEATURECONVERT 'PARSING'
-- %default FEATURECONVERT 'HASHING'

register ../../../target/pig_udfs-0.0.1.jar;
define Random myorg.pig.evaluation.MyRandom('$RANDOMSEED');
training = load '$TRAINFILE' using myorg.pig.storage.SVMLightLoader() as (label: int, features: map[]);
training = foreach training generate label, features, Random() as random;
training = order training by random parallel $PARTITIONS;
training = foreach training generate label, features;

store training into '$OUTDIR' using myorg.pig.storage.FeaturesSVMDCDBuilder('$FEATUREBIT', '$FEATURECONVERT', 'SQUAREDHINGE', '1.0');

-- store training into 'model2/' using myorg.pig.storage.FeaturesSVMDCDBuilder('model/');
