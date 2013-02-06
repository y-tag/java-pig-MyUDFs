package myorg.common;

public class PACommon {

    public enum PAType {
        PA, PA1, PA2, PALOG
    }

    public static EtaCalculatorPA getEtaCalculator(PAType paType) {
        switch (paType) {
            case PA1:
                return new EtaCalculatorPA1();
            case PA2:
                return new EtaCalculatorPA2();
            case PALOG:
                return new EtaCalculatorPALog();
            default:
                return new EtaCalculatorPA();
        }
    }

    public static class EtaCalculatorPA {
        public float calc(float loss, float squared_norm, float C) {
            return loss / squared_norm;
        }
    }

    public static class EtaCalculatorPA1 extends EtaCalculatorPA {
        @Override
        public float calc(float loss, float squared_norm, float C) {
            float eta = loss / squared_norm;
            if (C < eta) {
                eta = C;
            }
            return eta;
        }
    }

    public static class EtaCalculatorPA2 extends EtaCalculatorPA {
        @Override
        public float calc(float loss, float squared_norm, float C) {
            return loss / (squared_norm + (0.5f / C));
        }
    }

    // Algorithm 5 in
    // Yu et al., Dual Coordinate Descent Methods for Logistic Regression and Maximum Entropy Models
    public static class EtaCalculatorPALog extends EtaCalculatorPA {
        @Override
        public float calc(float loss, float squared_norm, float C) {
            float c1 = 1.0e-8f;
            float c2 = C - c1;
            float s = C; // c1 + c2
            float a = squared_norm;
            float b = 1.0f - loss; // y * predicted_value

            float zm = (c2 - c1) / 2.0f;

            float xi0 = 0.1f;
            float xi  = 0.1f;

            float zt = c1;
            float bt = b;
            float ct = c1;
            if (zm >= -b / a) {
                if (c1 >= s / 2) {
                    zt = xi0 * c1;
                }
            } else {
                if (c2 >= s / 2) {
                    zt = xi0 * c2;
                } else {
                    zt = c2;
                }
                bt = -b;
                ct = c2;
            }

            float epsilon = 1e-8f;

            for (int i = 0; i < 100; i++) {
                float g1 = (float)Math.log(zt / (s - zt)) + a * (zt - ct) + bt;
                if (Math.abs(g1) <= epsilon) {
                    break;
                }

                float g2 = a + (s / (zt * (s - zt)));

                float zt_d = zt - g1 / g2;
                if (zt_d <= 0.0f) {
                    zt = xi * zt;
                } else {
                    zt = zt_d;
                }
            }

            if (zm >= -b / a) {
                return zt;
            } else {
                return s - zt;
            }
        }
    }

}
