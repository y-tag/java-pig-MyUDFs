package myorg.classifier.online.binary;

import java.util.Map;
import java.util.HashMap;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.Assert.*;

import static org.junit.Assert.assertEquals;

import myorg.common.Convert;
import myorg.common.PACommon;
import myorg.classifier.online.binary.PA;
 
public class PATest {

    private static final float delta = 1e-5f;

    @BeforeClass
    public static void setUpClass() throws Exception {
    }

    @AfterClass
    public static void tearDownClass() throws Exception {
    }
 
    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }
 
    @Test
    public void testUpdateAndPredict() {
        Map<String, Float> features = new HashMap<String, Float>();

        // true weights
        // 1:0.5 2:2.0 3:-3.0

        float predicted_value = 0.0f;
        float expected_value  = 0.0f;

        int feature_bit = 5;
        float C = 1.0f;
        PA pa = new PA(feature_bit, Convert.FeatureConvert.PARSING, PACommon.PAType.PA, C);

        // 0, weights are all zero
        // y:+1 1:1.0
        features.clear();
        features.put("1", 1.0f);
        predicted_value = pa.predict(features);
        expected_value  = 0.0f;
        assertEquals(expected_value, predicted_value, delta);
        pa.updateWithPredictedValue(+1, features, predicted_value);
        // loss = 1.0 - (+1.0*0.0) = 1.0
        // squared_x_norm = 1.0**2 = 1.0
        // eta = 1.0 / 1.0 = 1.0
        // 1 ->  0.0 + 1.0 * +1.0 * 1.0 = 1.0

        // y:-1 3:1.0 1:0.5
        features.clear();
        features.put("3", 1.0f);
        features.put("1", 0.5f);
        predicted_value = pa.predict(features);
        expected_value = 1.0f * 0.5f;
        assertEquals(expected_value, predicted_value, delta);
        pa.update(-1, features);
        // loss = 1.0 - (-1.0*0.5) = 1.5
        // squared_x_norm = 1.0**2 + 0.5**2 = 1.25
        // eta = 1.5 / 1.25 = 1.2
        // 1 ->  1.0 + 1.2 * -1.0 * 0.5 =  0.4
        // 3 ->  0.0 + 1.2 * -1.0 * 1.0 = -1.2

        // y:-1 3:0.5 1:2.0 2:-1.0
        features.clear();
        features.put("3",  0.5f);
        features.put("1",  2.0f);
        features.put("2", -1.0f);
        predicted_value = pa.predict(features);
        expected_value = 0.4f * 2.0f + -1.2f * 0.5f;
        assertEquals(expected_value, predicted_value, delta);
        pa.updateWithPredictedValue(-1, features, predicted_value);
        // loss = 1.0 - (-1.0*0.2) = 1.2
        // squared_x_norm = 0.5**2 + 2.0**2 + -1.0**2 = 5.25
        // eta = 1.2 / 5.25
        // 1 ->  0.4 + (1.2 / 5.25) * -1.0 *  2.0
        // 2 ->  0.0 + (1.2 / 5.25) * -1.0 * -1.0
        // 3 -> -1.2 + (1.2 / 5.25) * -1.0 *  0.5

        // 3, weights-> 1:1.0 2:0.5
        // y:+1 1:1.0 2:0.5
        features.clear();
        features.put("1", 1.0f);
        features.put("2", 0.5f);
        predicted_value = pa.predict(features);
        expected_value = ( 0.4f + (1.2f / 5.25f) * -1.0f *  2.0f) * 1.0f +
                         ( 0.0f + (1.2f / 5.25f) * -1.0f * -1.0f) * 0.5f;
        assertEquals(expected_value, predicted_value, delta);
        pa.update(+1, features);
        // loss = 1.0 - (-1.0*0.2) = 1.2
        // squared_x_norm = 0.5**2 + 2.0**2 + -1.0**2 = 5.25
        // eta = 1.2 / 5.25
        // 1 ->  1.0 + (1.2 / 5.25) * -1.0 *  2.0
        // 2 ->  0.0 + (1.2 / 5.25) * -1.0 * -1.0
        // 3 ->  0.5 + (1.2 / 5.25) * -1.0 *  0.5

    }
 
}
