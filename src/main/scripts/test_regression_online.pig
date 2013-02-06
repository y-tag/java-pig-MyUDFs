-- test.pig
%default TESTFILE 'E2006.test.gz'
%default MODELDIR 'model/'

register ../../../target/pig_udfs-0.0.1.jar;
define Predict myorg.pig.evaluation.PredictWithOnlineRegression('$MODELDIR');
test = load '$TESTFILE' using myorg.pig.storage.SVMLightLoader() as (target: float, features: map[]);
predict = foreach test generate target, Predict(features) as prediction;

-- store predict into 'result/';

result = foreach predict generate (target-prediction)*(target-prediction) as se;
result = group result all;
result = foreach result generate AVG(result.se) as mse;
dump result;
