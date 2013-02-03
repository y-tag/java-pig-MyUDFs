package myudfs;

import com.google.common.hash.Hashing;

public class Convert {

    enum FeatureConvert {
        PARSING, HASHING
    }

    static public StrToIntConverter getStrToIntConverter(FeatureConvert convertType) {
        switch (convertType) {
            case PARSING:
                return new StrToIntConverterWithParsing();
            case HASHING:
                return new StrToIntConverterWithHashing();
            default:
                return new StrToIntConverterWithHashing();
        }
    }

    interface StrToIntConverter {
        int convert(String str);
    }

    public static class StrToIntConverterWithParsing implements StrToIntConverter {
        public int convert(String str) {
            return Integer.parseInt(str);
        }
    }

    public static class StrToIntConverterWithHashing implements StrToIntConverter {
        public int convert(String str) {
            return Hashing.murmur3_32().hashString(str).asInt();
        }
    }

}
