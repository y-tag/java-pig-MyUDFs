-- train.pig
%default RANDOMSEED '1000'
%default PARTITIONS '3'
%default TRAINFILE  'kddcup2012_trak2_training.txt.gz'
%default OUTDIR     'model/'
%default FEATUREBIT '20'
%default FEATURECONVERT 'HASHING'

register ../../../target/pig_udfs-0.0.1.jar;
define Random myorg.pig.evaluation.MyRandom('$RANDOMSEED');
training = load '$TRAINFILE' as (click:int, imp:int, url:chararray, ad:chararray, advertiser:chararray, depth:chararray, position:chararray, query:chararray, keyword:chararray, title:chararray, description:chararray, user:chararray);
training = foreach training generate click, imp, TOMAP(CONCAT('url_',url),1.0f, CONCAT('ad_',ad),1.0f, CONCAT('advertiser_',advertiser),1.0f, CONCAT('depth_',depth),1.0f, CONCAT('position_',position),1.0f, CONCAT('query_',query),1.0f, CONCAT('keyword_',keyword),1.0f, CONCAT('title_',title),1.0f, CONCAT('description_',description),1.0f, CONCAT('user_',user),1.0f) as features;
training = foreach training generate (click > 0 ? 1 : -1) as label, features;
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
