-- test.pig
%default TESTFILE  'kddcup2012_trak2_test.txt'
%default MODELDIR  'model/'
%default RESULTDIR 'result/'

register ../../../target/pig_udfs-0.0.1.jar;
define Classify myorg.pig.evaluation.ClassifyWithBinaryOnlineClassifier('$MODELDIR');
-- define Classify myorg.pig.evaluation.ClassifyWithBinaryOnlineClassifierVoting('$MODELDIR');
test = load '$TESTFILE' as (url:chararray, ad:chararray, advertiser:chararray, depth:chararray, position:chararray, query:chararray, keyword:chararray, title:chararray, description:chararray, user:chararray);
test = foreach test generate TOMAP(CONCAT('url_',url),1.0f, CONCAT('ad_',ad),1.0f, CONCAT('advertiser_',advertiser),1.0f, CONCAT('depth_',depth),1.0f, CONCAT('position_',position),1.0f, CONCAT('query_',query),1.0f, CONCAT('keyword_',keyword),1.0f, CONCAT('title_',title),1.0f, CONCAT('description_',description),1.0f, CONCAT('user_',user),1.0f) as features;
predict = foreach test generate Classify(features) as prediction;
store predict into '$RESULTDIR';
