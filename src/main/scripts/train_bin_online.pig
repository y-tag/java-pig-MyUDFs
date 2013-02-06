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

-- store training into '$OUTDIR' using myorg.pig.storage.FeaturesPerceptronBuilder('$FEATUREBIT', '$FEATURECONVERT');
store training into '$OUTDIR' using myorg.pig.storage.FeaturesPABuilder('$FEATUREBIT', '$FEATURECONVERT', 'PA2', '1.0');
-- store training into '$OUTDIR' using myorg.pig.storage.FeaturesSVMSGDBuilder('$FEATUREBIT', '$FEATURECONVERT', 'SQUAREDHINGE', '1.0', '10');
-- store training into '$OUTDIR' using myorg.pig.storage.FeaturesPegasosBuilder('$FEATUREBIT', '$FEATURECONVERT', 'HINGE', '10.0', '32');

-- store training into 'model2/' using myorg.pig.storage.FeaturesPerceptronBuilder('model/');
-- store training into 'model2/' using myorg.pig.storage.FeaturesPABuilder('model/');
-- store training into 'model2/' using myorg.pig.storage.FeaturesSVMSGDBuilder('model/');
