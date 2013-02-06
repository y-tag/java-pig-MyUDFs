-- test.pig
%default TESTFILE 'rcv1_test.multiclass.gz'
%default MODELDIR 'model/'

register ../../../target/pig_udfs-0.0.1.jar;
define Classify myorg.pig.evaluation.ClassifyWithMulticlassOnlineClassifier('$MODELDIR');
test = load '$TESTFILE' using myorg.pig.storage.SVMLightLoader() as (label: chararray, features: map[]);
-- test = limit test 10000;
predict = foreach test generate label, Classify(features) as prediction;
results = foreach predict generate (label == prediction ? 1 : 0) as matching;
cnt = group results by matching;
cnt = foreach cnt generate group, COUNT(results);
dump cnt;
