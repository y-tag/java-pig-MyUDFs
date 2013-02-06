-- test.pig
%default TESTFILE 'rcv1_test.binary.gz'
%default MODELDIR 'model/'

register ../../../target/pig_udfs-0.0.1.jar;
define Classify myorg.pig.evaluation.ClassifyWithBinaryBatchClassifier('$MODELDIR');
test = load '$TESTFILE' using myorg.pig.storage.SVMLightLoader() as (label: int, features: map[]);
predict = foreach test generate label, (Classify(features) > 0 ? 1 : -1) as prediction;
results = foreach predict generate (label == prediction ? 1 : 0) as matching;
cnt = group results by matching;
cnt = foreach cnt generate group, COUNT(results);
dump cnt;
