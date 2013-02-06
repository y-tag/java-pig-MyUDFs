-- train.pig
%default RANDOMSEED '1000'
%default PARTITIONS '3'
%default TRAINFILE  'E2006.train.gz'
%default OUTDIR     'model/'
%default FEATUREBIT '20'
%default FEATURECONVERT 'PARSING'
-- %default FEATURECONVERT 'HASHING'

register ../../../target/pig_udfs-0.0.1.jar;
define Random myorg.pig.evaluation.MyRandom('$RANDOMSEED');
training = load '$TRAINFILE' using myorg.pig.storage.SVMLightLoader() as (target: float, features: map[]);
training = foreach training generate target, features, Random() as random;
training = order training by random parallel $PARTITIONS;
training = foreach training generate target, features;

store training into '$OUTDIR' using myorg.pig.storage.FeaturesRegressionPABuilder('$FEATUREBIT', '$FEATURECONVERT', 'PA2', '1.0', '0.1');

-- store training into 'model2/' using myorg.pig.storage.FeaturesRegressionPABuilder('model/');
