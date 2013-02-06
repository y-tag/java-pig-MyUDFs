-- train.pig
%default RANDOMSEED '1000'
%default PARTITIONS '3'
%default TRAINFILE  'rcv1_train.multiclass.gz'
%default OUTDIR     'model/'
%default FEATUREBIT '20'
%default FEATURECONVERT 'PARSING'
-- %default FEATURECONVERT 'HASHING'

register ../../../target/pig_udfs-0.0.1.jar;
define Random myorg.pig.evaluation.MyRandom('$RANDOMSEED');
training = load '$TRAINFILE' using myorg.pig.storage.SVMLightLoader() as (label: chararray, features: map[]);
training = foreach training generate label, features, Random() as random;
training = order training by random parallel $PARTITIONS;
training = foreach training generate label, features;

-- store training into '$OUTDIR' using myorg.pig.storage.FeaturesMulticlassPerceptronBuilder('$FEATUREBIT', '$FEATURECONVERT');
store training into '$OUTDIR' using myorg.pig.storage.FeaturesMulticlassPABuilder('$FEATUREBIT', '$FEATURECONVERT', 'PA2', '1.0');

-- store training into 'model2/' using myorg.pig.storage.FeaturesPerceptronBuilder('model/');
-- store training into 'model2/' using myorg.pig.storage.FeaturesPABuilder('model/');
