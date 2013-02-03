-- train.pig
%default RANDOMSEED '1000'
%default PARTITIONS '3'
%default TRAINFILE  'rcv1_train.binary.gz'
%default OUTDIR     'model/'
%default FEATUREBIT '20'
%default FEATURECONVERT 'PARSING'
-- %default FEATURECONVERT 'HASHING'

register ../../../target/pig_udfs-0.0.1.jar;
define Random myudfs.MyRandom('$RANDOMSEED');
training = load '$TRAINFILE' using myudfs.SVMLightLoader() as (label: int, features: map[]);
training = foreach training generate label, features, Random() as random;
training = order training by random parallel $PARTITIONS;
training = foreach training generate label, features;

-- store training into '$OUTDIR' using myudfs.FeaturesPerceptronBuilder('$FEATUREBIT', '$FEATURECONVERT');
store training into '$OUTDIR' using myudfs.FeaturesPABuilder('$FEATUREBIT', '$FEATURECONVERT', 'PA2', '1.0');
-- store training into '$OUTDIR' using myudfs.FeaturesSVMSGDBuilder('$FEATUREBIT', '$FEATURECONVERT', 'SQUAREDHINGE', '1.0', '10');
-- store training into '$OUTDIR' using myudfs.FeaturesPegasosBuilder('$FEATUREBIT', '$FEATURECONVERT', 'HINGE', '10.0', '32');

-- store training into 'model2/' using myudfs.FeaturesPerceptronBuilder('model/');
-- store training into 'model2/' using myudfs.FeaturesPABuilder('model/');
-- store training into 'model2/' using myudfs.FeaturesSVMSGDBuilder('model/');
