/**
* Copyright (c) 2026 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
/**
* Copyright (c) 2020 Intel Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once

#include "activation.hpp"

namespace tinymind {
    #if TINYMIND_USE_SIN_1_127
    struct SinValuesTableQ1_127
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_1_127
    #if TINYMIND_USE_SIN_2_126
    struct SinValuesTableQ2_126
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_2_126
    #if TINYMIND_USE_SIN_3_125
    struct SinValuesTableQ3_125
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_3_125
    #if TINYMIND_USE_SIN_4_124
    struct SinValuesTableQ4_124
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_4_124
    #if TINYMIND_USE_SIN_5_123
    struct SinValuesTableQ5_123
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_5_123
    #if TINYMIND_USE_SIN_6_122
    struct SinValuesTableQ6_122
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_6_122
    #if TINYMIND_USE_SIN_7_121
    struct SinValuesTableQ7_121
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_7_121
    #if TINYMIND_USE_SIN_8_120
    struct SinValuesTableQ8_120
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_8_120
    #if TINYMIND_USE_SIN_9_119
    struct SinValuesTableQ9_119
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_9_119
    #if TINYMIND_USE_SIN_10_118
    struct SinValuesTableQ10_118
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_10_118
    #if TINYMIND_USE_SIN_11_117
    struct SinValuesTableQ11_117
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_11_117
    #if TINYMIND_USE_SIN_12_116
    struct SinValuesTableQ12_116
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_12_116
    #if TINYMIND_USE_SIN_13_115
    struct SinValuesTableQ13_115
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_13_115
    #if TINYMIND_USE_SIN_14_114
    struct SinValuesTableQ14_114
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_14_114
    #if TINYMIND_USE_SIN_15_113
    struct SinValuesTableQ15_113
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_15_113
    #if TINYMIND_USE_SIN_16_112
    struct SinValuesTableQ16_112
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_16_112
    #if TINYMIND_USE_SIN_17_111
    struct SinValuesTableQ17_111
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_17_111
    #if TINYMIND_USE_SIN_18_110
    struct SinValuesTableQ18_110
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_18_110
    #if TINYMIND_USE_SIN_19_109
    struct SinValuesTableQ19_109
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_19_109
    #if TINYMIND_USE_SIN_20_108
    struct SinValuesTableQ20_108
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_20_108
    #if TINYMIND_USE_SIN_21_107
    struct SinValuesTableQ21_107
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_21_107
    #if TINYMIND_USE_SIN_22_106
    struct SinValuesTableQ22_106
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_22_106
    #if TINYMIND_USE_SIN_23_105
    struct SinValuesTableQ23_105
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_23_105
    #if TINYMIND_USE_SIN_24_104
    struct SinValuesTableQ24_104
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_24_104
    #if TINYMIND_USE_SIN_25_103
    struct SinValuesTableQ25_103
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_25_103
    #if TINYMIND_USE_SIN_26_102
    struct SinValuesTableQ26_102
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_26_102
    #if TINYMIND_USE_SIN_27_101
    struct SinValuesTableQ27_101
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_27_101
    #if TINYMIND_USE_SIN_28_100
    struct SinValuesTableQ28_100
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_28_100
    #if TINYMIND_USE_SIN_29_99
    struct SinValuesTableQ29_99
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_29_99
    #if TINYMIND_USE_SIN_30_98
    struct SinValuesTableQ30_98
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_30_98
    #if TINYMIND_USE_SIN_31_97
    struct SinValuesTableQ31_97
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_31_97
    #if TINYMIND_USE_SIN_32_96
    struct SinValuesTableQ32_96
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_32_96
    #if TINYMIND_USE_SIN_33_95
    struct SinValuesTableQ33_95
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_33_95
    #if TINYMIND_USE_SIN_34_94
    struct SinValuesTableQ34_94
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_34_94
    #if TINYMIND_USE_SIN_35_93
    struct SinValuesTableQ35_93
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_35_93
    #if TINYMIND_USE_SIN_36_92
    struct SinValuesTableQ36_92
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_36_92
    #if TINYMIND_USE_SIN_37_91
    struct SinValuesTableQ37_91
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_37_91
    #if TINYMIND_USE_SIN_38_90
    struct SinValuesTableQ38_90
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_38_90
    #if TINYMIND_USE_SIN_39_89
    struct SinValuesTableQ39_89
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_39_89
    #if TINYMIND_USE_SIN_40_88
    struct SinValuesTableQ40_88
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_40_88
    #if TINYMIND_USE_SIN_41_87
    struct SinValuesTableQ41_87
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_41_87
    #if TINYMIND_USE_SIN_42_86
    struct SinValuesTableQ42_86
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_42_86
    #if TINYMIND_USE_SIN_43_85
    struct SinValuesTableQ43_85
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_43_85
    #if TINYMIND_USE_SIN_44_84
    struct SinValuesTableQ44_84
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_44_84
    #if TINYMIND_USE_SIN_45_83
    struct SinValuesTableQ45_83
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_45_83
    #if TINYMIND_USE_SIN_46_82
    struct SinValuesTableQ46_82
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_46_82
    #if TINYMIND_USE_SIN_47_81
    struct SinValuesTableQ47_81
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_47_81
    #if TINYMIND_USE_SIN_48_80
    struct SinValuesTableQ48_80
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_48_80
    #if TINYMIND_USE_SIN_49_79
    struct SinValuesTableQ49_79
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_49_79
    #if TINYMIND_USE_SIN_50_78
    struct SinValuesTableQ50_78
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_50_78
    #if TINYMIND_USE_SIN_51_77
    struct SinValuesTableQ51_77
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_51_77
    #if TINYMIND_USE_SIN_52_76
    struct SinValuesTableQ52_76
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_52_76
    #if TINYMIND_USE_SIN_53_75
    struct SinValuesTableQ53_75
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_53_75
    #if TINYMIND_USE_SIN_54_74
    struct SinValuesTableQ54_74
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_54_74
    #if TINYMIND_USE_SIN_55_73
    struct SinValuesTableQ55_73
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_55_73
    #if TINYMIND_USE_SIN_56_72
    struct SinValuesTableQ56_72
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_56_72
    #if TINYMIND_USE_SIN_57_71
    struct SinValuesTableQ57_71
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_57_71
    #if TINYMIND_USE_SIN_58_70
    struct SinValuesTableQ58_70
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_58_70
    #if TINYMIND_USE_SIN_59_69
    struct SinValuesTableQ59_69
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_59_69
    #if TINYMIND_USE_SIN_60_68
    struct SinValuesTableQ60_68
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_60_68
    #if TINYMIND_USE_SIN_61_67
    struct SinValuesTableQ61_67
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_61_67
    #if TINYMIND_USE_SIN_62_66
    struct SinValuesTableQ62_66
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_62_66
    #if TINYMIND_USE_SIN_63_65
    struct SinValuesTableQ63_65
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_63_65
    #if TINYMIND_USE_SIN_64_64
    struct SinValuesTableQ64_64
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_64_64
    #if TINYMIND_USE_SIN_65_63
    struct SinValuesTableQ65_63
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_65_63
    #if TINYMIND_USE_SIN_66_62
    struct SinValuesTableQ66_62
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_66_62
    #if TINYMIND_USE_SIN_67_61
    struct SinValuesTableQ67_61
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_67_61
    #if TINYMIND_USE_SIN_68_60
    struct SinValuesTableQ68_60
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_68_60
    #if TINYMIND_USE_SIN_69_59
    struct SinValuesTableQ69_59
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_69_59
    #if TINYMIND_USE_SIN_70_58
    struct SinValuesTableQ70_58
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_70_58
    #if TINYMIND_USE_SIN_71_57
    struct SinValuesTableQ71_57
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_71_57
    #if TINYMIND_USE_SIN_72_56
    struct SinValuesTableQ72_56
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_72_56
    #if TINYMIND_USE_SIN_73_55
    struct SinValuesTableQ73_55
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_73_55
    #if TINYMIND_USE_SIN_74_54
    struct SinValuesTableQ74_54
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_74_54
    #if TINYMIND_USE_SIN_75_53
    struct SinValuesTableQ75_53
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_75_53
    #if TINYMIND_USE_SIN_76_52
    struct SinValuesTableQ76_52
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_76_52
    #if TINYMIND_USE_SIN_77_51
    struct SinValuesTableQ77_51
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_77_51
    #if TINYMIND_USE_SIN_78_50
    struct SinValuesTableQ78_50
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_78_50
    #if TINYMIND_USE_SIN_79_49
    struct SinValuesTableQ79_49
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_79_49
    #if TINYMIND_USE_SIN_80_48
    struct SinValuesTableQ80_48
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_80_48
    #if TINYMIND_USE_SIN_81_47
    struct SinValuesTableQ81_47
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_81_47
    #if TINYMIND_USE_SIN_82_46
    struct SinValuesTableQ82_46
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_82_46
    #if TINYMIND_USE_SIN_83_45
    struct SinValuesTableQ83_45
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_83_45
    #if TINYMIND_USE_SIN_84_44
    struct SinValuesTableQ84_44
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_84_44
    #if TINYMIND_USE_SIN_85_43
    struct SinValuesTableQ85_43
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_85_43
    #if TINYMIND_USE_SIN_86_42
    struct SinValuesTableQ86_42
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_86_42
    #if TINYMIND_USE_SIN_87_41
    struct SinValuesTableQ87_41
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_87_41
    #if TINYMIND_USE_SIN_88_40
    struct SinValuesTableQ88_40
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_88_40
    #if TINYMIND_USE_SIN_89_39
    struct SinValuesTableQ89_39
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_89_39
    #if TINYMIND_USE_SIN_90_38
    struct SinValuesTableQ90_38
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_90_38
    #if TINYMIND_USE_SIN_91_37
    struct SinValuesTableQ91_37
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_91_37
    #if TINYMIND_USE_SIN_92_36
    struct SinValuesTableQ92_36
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_92_36
    #if TINYMIND_USE_SIN_93_35
    struct SinValuesTableQ93_35
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_93_35
    #if TINYMIND_USE_SIN_94_34
    struct SinValuesTableQ94_34
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_94_34
    #if TINYMIND_USE_SIN_95_33
    struct SinValuesTableQ95_33
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_95_33
    #if TINYMIND_USE_SIN_96_32
    struct SinValuesTableQ96_32
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_96_32
    #if TINYMIND_USE_SIN_97_31
    struct SinValuesTableQ97_31
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_97_31
    #if TINYMIND_USE_SIN_98_30
    struct SinValuesTableQ98_30
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_98_30
    #if TINYMIND_USE_SIN_99_29
    struct SinValuesTableQ99_29
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_99_29
    #if TINYMIND_USE_SIN_100_28
    struct SinValuesTableQ100_28
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_100_28
    #if TINYMIND_USE_SIN_101_27
    struct SinValuesTableQ101_27
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_101_27
    #if TINYMIND_USE_SIN_102_26
    struct SinValuesTableQ102_26
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_102_26
    #if TINYMIND_USE_SIN_103_25
    struct SinValuesTableQ103_25
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_103_25
    #if TINYMIND_USE_SIN_104_24
    struct SinValuesTableQ104_24
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_104_24
    #if TINYMIND_USE_SIN_105_23
    struct SinValuesTableQ105_23
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_105_23
    #if TINYMIND_USE_SIN_106_22
    struct SinValuesTableQ106_22
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_106_22
    #if TINYMIND_USE_SIN_107_21
    struct SinValuesTableQ107_21
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_107_21
    #if TINYMIND_USE_SIN_108_20
    struct SinValuesTableQ108_20
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_108_20
    #if TINYMIND_USE_SIN_109_19
    struct SinValuesTableQ109_19
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_109_19
    #if TINYMIND_USE_SIN_110_18
    struct SinValuesTableQ110_18
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_110_18
    #if TINYMIND_USE_SIN_111_17
    struct SinValuesTableQ111_17
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_111_17
    #if TINYMIND_USE_SIN_112_16
    struct SinValuesTableQ112_16
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_112_16
    #if TINYMIND_USE_SIN_113_15
    struct SinValuesTableQ113_15
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_113_15
    #if TINYMIND_USE_SIN_114_14
    struct SinValuesTableQ114_14
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_114_14
    #if TINYMIND_USE_SIN_115_13
    struct SinValuesTableQ115_13
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_115_13
    #if TINYMIND_USE_SIN_116_12
    struct SinValuesTableQ116_12
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_116_12
    #if TINYMIND_USE_SIN_117_11
    struct SinValuesTableQ117_11
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_117_11
    #if TINYMIND_USE_SIN_118_10
    struct SinValuesTableQ118_10
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_118_10
    #if TINYMIND_USE_SIN_119_9
    struct SinValuesTableQ119_9
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_119_9
    #if TINYMIND_USE_SIN_120_8
    struct SinValuesTableQ120_8
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_120_8
    #if TINYMIND_USE_SIN_121_7
    struct SinValuesTableQ121_7
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_121_7
    #if TINYMIND_USE_SIN_122_6
    struct SinValuesTableQ122_6
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_122_6
    #if TINYMIND_USE_SIN_123_5
    struct SinValuesTableQ123_5
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_123_5
    #if TINYMIND_USE_SIN_124_4
    struct SinValuesTableQ124_4
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_124_4
    #if TINYMIND_USE_SIN_125_3
    struct SinValuesTableQ125_3
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_125_3
    #if TINYMIND_USE_SIN_126_2
    struct SinValuesTableQ126_2
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_126_2
    #if TINYMIND_USE_SIN_127_1
    struct SinValuesTableQ127_1
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_127_1
}
