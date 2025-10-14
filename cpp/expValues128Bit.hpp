/**
* Copyright (c) 2023 Dan McLeran
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
    #if TINYMIND_USE_EXP_1_127
    struct ExpValuesTableQ1_127
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_1_127
    #if TINYMIND_USE_EXP_2_126
    struct ExpValuesTableQ2_126
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_2_126
    #if TINYMIND_USE_EXP_3_125
    struct ExpValuesTableQ3_125
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_3_125
    #if TINYMIND_USE_EXP_4_124
    struct ExpValuesTableQ4_124
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_4_124
    #if TINYMIND_USE_EXP_5_123
    struct ExpValuesTableQ5_123
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_5_123
    #if TINYMIND_USE_EXP_6_122
    struct ExpValuesTableQ6_122
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_6_122
    #if TINYMIND_USE_EXP_7_121
    struct ExpValuesTableQ7_121
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_7_121
    #if TINYMIND_USE_EXP_8_120
    struct ExpValuesTableQ8_120
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_8_120
    #if TINYMIND_USE_EXP_9_119
    struct ExpValuesTableQ9_119
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_9_119
    #if TINYMIND_USE_EXP_10_118
    struct ExpValuesTableQ10_118
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_10_118
    #if TINYMIND_USE_EXP_11_117
    struct ExpValuesTableQ11_117
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_11_117
    #if TINYMIND_USE_EXP_12_116
    struct ExpValuesTableQ12_116
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_12_116
    #if TINYMIND_USE_EXP_13_115
    struct ExpValuesTableQ13_115
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_13_115
    #if TINYMIND_USE_EXP_14_114
    struct ExpValuesTableQ14_114
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_14_114
    #if TINYMIND_USE_EXP_15_113
    struct ExpValuesTableQ15_113
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_15_113
    #if TINYMIND_USE_EXP_16_112
    struct ExpValuesTableQ16_112
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_16_112
    #if TINYMIND_USE_EXP_17_111
    struct ExpValuesTableQ17_111
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_17_111
    #if TINYMIND_USE_EXP_18_110
    struct ExpValuesTableQ18_110
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_18_110
    #if TINYMIND_USE_EXP_19_109
    struct ExpValuesTableQ19_109
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_19_109
    #if TINYMIND_USE_EXP_20_108
    struct ExpValuesTableQ20_108
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_20_108
    #if TINYMIND_USE_EXP_21_107
    struct ExpValuesTableQ21_107
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_21_107
    #if TINYMIND_USE_EXP_22_106
    struct ExpValuesTableQ22_106
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_22_106
    #if TINYMIND_USE_EXP_23_105
    struct ExpValuesTableQ23_105
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_23_105
    #if TINYMIND_USE_EXP_24_104
    struct ExpValuesTableQ24_104
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_24_104
    #if TINYMIND_USE_EXP_25_103
    struct ExpValuesTableQ25_103
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_25_103
    #if TINYMIND_USE_EXP_26_102
    struct ExpValuesTableQ26_102
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_26_102
    #if TINYMIND_USE_EXP_27_101
    struct ExpValuesTableQ27_101
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_27_101
    #if TINYMIND_USE_EXP_28_100
    struct ExpValuesTableQ28_100
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_28_100
    #if TINYMIND_USE_EXP_29_99
    struct ExpValuesTableQ29_99
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_29_99
    #if TINYMIND_USE_EXP_30_98
    struct ExpValuesTableQ30_98
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_30_98
    #if TINYMIND_USE_EXP_31_97
    struct ExpValuesTableQ31_97
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_31_97
    #if TINYMIND_USE_EXP_32_96
    struct ExpValuesTableQ32_96
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_32_96
    #if TINYMIND_USE_EXP_33_95
    struct ExpValuesTableQ33_95
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_33_95
    #if TINYMIND_USE_EXP_34_94
    struct ExpValuesTableQ34_94
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_34_94
    #if TINYMIND_USE_EXP_35_93
    struct ExpValuesTableQ35_93
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_35_93
    #if TINYMIND_USE_EXP_36_92
    struct ExpValuesTableQ36_92
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_36_92
    #if TINYMIND_USE_EXP_37_91
    struct ExpValuesTableQ37_91
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_37_91
    #if TINYMIND_USE_EXP_38_90
    struct ExpValuesTableQ38_90
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_38_90
    #if TINYMIND_USE_EXP_39_89
    struct ExpValuesTableQ39_89
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_39_89
    #if TINYMIND_USE_EXP_40_88
    struct ExpValuesTableQ40_88
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_40_88
    #if TINYMIND_USE_EXP_41_87
    struct ExpValuesTableQ41_87
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_41_87
    #if TINYMIND_USE_EXP_42_86
    struct ExpValuesTableQ42_86
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_42_86
    #if TINYMIND_USE_EXP_43_85
    struct ExpValuesTableQ43_85
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_43_85
    #if TINYMIND_USE_EXP_44_84
    struct ExpValuesTableQ44_84
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_44_84
    #if TINYMIND_USE_EXP_45_83
    struct ExpValuesTableQ45_83
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_45_83
    #if TINYMIND_USE_EXP_46_82
    struct ExpValuesTableQ46_82
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_46_82
    #if TINYMIND_USE_EXP_47_81
    struct ExpValuesTableQ47_81
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_47_81
    #if TINYMIND_USE_EXP_48_80
    struct ExpValuesTableQ48_80
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_48_80
    #if TINYMIND_USE_EXP_49_79
    struct ExpValuesTableQ49_79
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_49_79
    #if TINYMIND_USE_EXP_50_78
    struct ExpValuesTableQ50_78
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_50_78
    #if TINYMIND_USE_EXP_51_77
    struct ExpValuesTableQ51_77
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_51_77
    #if TINYMIND_USE_EXP_52_76
    struct ExpValuesTableQ52_76
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_52_76
    #if TINYMIND_USE_EXP_53_75
    struct ExpValuesTableQ53_75
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_53_75
    #if TINYMIND_USE_EXP_54_74
    struct ExpValuesTableQ54_74
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_54_74
    #if TINYMIND_USE_EXP_55_73
    struct ExpValuesTableQ55_73
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_55_73
    #if TINYMIND_USE_EXP_56_72
    struct ExpValuesTableQ56_72
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_56_72
    #if TINYMIND_USE_EXP_57_71
    struct ExpValuesTableQ57_71
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_57_71
    #if TINYMIND_USE_EXP_58_70
    struct ExpValuesTableQ58_70
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_58_70
    #if TINYMIND_USE_EXP_59_69
    struct ExpValuesTableQ59_69
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_59_69
    #if TINYMIND_USE_EXP_60_68
    struct ExpValuesTableQ60_68
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_60_68
    #if TINYMIND_USE_EXP_61_67
    struct ExpValuesTableQ61_67
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_61_67
    #if TINYMIND_USE_EXP_62_66
    struct ExpValuesTableQ62_66
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_62_66
    #if TINYMIND_USE_EXP_63_65
    struct ExpValuesTableQ63_65
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_63_65
    #if TINYMIND_USE_EXP_64_64
    struct ExpValuesTableQ64_64
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_64_64
    #if TINYMIND_USE_EXP_65_63
    struct ExpValuesTableQ65_63
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_65_63
    #if TINYMIND_USE_EXP_66_62
    struct ExpValuesTableQ66_62
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_66_62
    #if TINYMIND_USE_EXP_67_61
    struct ExpValuesTableQ67_61
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_67_61
    #if TINYMIND_USE_EXP_68_60
    struct ExpValuesTableQ68_60
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_68_60
    #if TINYMIND_USE_EXP_69_59
    struct ExpValuesTableQ69_59
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_69_59
    #if TINYMIND_USE_EXP_70_58
    struct ExpValuesTableQ70_58
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_70_58
    #if TINYMIND_USE_EXP_71_57
    struct ExpValuesTableQ71_57
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_71_57
    #if TINYMIND_USE_EXP_72_56
    struct ExpValuesTableQ72_56
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_72_56
    #if TINYMIND_USE_EXP_73_55
    struct ExpValuesTableQ73_55
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_73_55
    #if TINYMIND_USE_EXP_74_54
    struct ExpValuesTableQ74_54
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_74_54
    #if TINYMIND_USE_EXP_75_53
    struct ExpValuesTableQ75_53
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_75_53
    #if TINYMIND_USE_EXP_76_52
    struct ExpValuesTableQ76_52
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_76_52
    #if TINYMIND_USE_EXP_77_51
    struct ExpValuesTableQ77_51
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_77_51
    #if TINYMIND_USE_EXP_78_50
    struct ExpValuesTableQ78_50
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_78_50
    #if TINYMIND_USE_EXP_79_49
    struct ExpValuesTableQ79_49
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_79_49
    #if TINYMIND_USE_EXP_80_48
    struct ExpValuesTableQ80_48
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_80_48
    #if TINYMIND_USE_EXP_81_47
    struct ExpValuesTableQ81_47
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_81_47
    #if TINYMIND_USE_EXP_82_46
    struct ExpValuesTableQ82_46
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_82_46
    #if TINYMIND_USE_EXP_83_45
    struct ExpValuesTableQ83_45
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_83_45
    #if TINYMIND_USE_EXP_84_44
    struct ExpValuesTableQ84_44
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_84_44
    #if TINYMIND_USE_EXP_85_43
    struct ExpValuesTableQ85_43
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_85_43
    #if TINYMIND_USE_EXP_86_42
    struct ExpValuesTableQ86_42
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_86_42
    #if TINYMIND_USE_EXP_87_41
    struct ExpValuesTableQ87_41
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_87_41
    #if TINYMIND_USE_EXP_88_40
    struct ExpValuesTableQ88_40
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_88_40
    #if TINYMIND_USE_EXP_89_39
    struct ExpValuesTableQ89_39
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_89_39
    #if TINYMIND_USE_EXP_90_38
    struct ExpValuesTableQ90_38
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_90_38
    #if TINYMIND_USE_EXP_91_37
    struct ExpValuesTableQ91_37
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_91_37
    #if TINYMIND_USE_EXP_92_36
    struct ExpValuesTableQ92_36
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_92_36
    #if TINYMIND_USE_EXP_93_35
    struct ExpValuesTableQ93_35
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_93_35
    #if TINYMIND_USE_EXP_94_34
    struct ExpValuesTableQ94_34
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_94_34
    #if TINYMIND_USE_EXP_95_33
    struct ExpValuesTableQ95_33
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_95_33
    #if TINYMIND_USE_EXP_96_32
    struct ExpValuesTableQ96_32
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_96_32
    #if TINYMIND_USE_EXP_97_31
    struct ExpValuesTableQ97_31
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_97_31
    #if TINYMIND_USE_EXP_98_30
    struct ExpValuesTableQ98_30
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_98_30
    #if TINYMIND_USE_EXP_99_29
    struct ExpValuesTableQ99_29
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_99_29
    #if TINYMIND_USE_EXP_100_28
    struct ExpValuesTableQ100_28
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_100_28
    #if TINYMIND_USE_EXP_101_27
    struct ExpValuesTableQ101_27
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_101_27
    #if TINYMIND_USE_EXP_102_26
    struct ExpValuesTableQ102_26
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_102_26
    #if TINYMIND_USE_EXP_103_25
    struct ExpValuesTableQ103_25
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_103_25
    #if TINYMIND_USE_EXP_104_24
    struct ExpValuesTableQ104_24
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_104_24
    #if TINYMIND_USE_EXP_105_23
    struct ExpValuesTableQ105_23
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_105_23
    #if TINYMIND_USE_EXP_106_22
    struct ExpValuesTableQ106_22
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_106_22
    #if TINYMIND_USE_EXP_107_21
    struct ExpValuesTableQ107_21
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_107_21
    #if TINYMIND_USE_EXP_108_20
    struct ExpValuesTableQ108_20
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_108_20
    #if TINYMIND_USE_EXP_109_19
    struct ExpValuesTableQ109_19
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_109_19
    #if TINYMIND_USE_EXP_110_18
    struct ExpValuesTableQ110_18
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_110_18
    #if TINYMIND_USE_EXP_111_17
    struct ExpValuesTableQ111_17
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_111_17
    #if TINYMIND_USE_EXP_112_16
    struct ExpValuesTableQ112_16
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_112_16
    #if TINYMIND_USE_EXP_113_15
    struct ExpValuesTableQ113_15
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_113_15
    #if TINYMIND_USE_EXP_114_14
    struct ExpValuesTableQ114_14
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_114_14
    #if TINYMIND_USE_EXP_115_13
    struct ExpValuesTableQ115_13
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_115_13
    #if TINYMIND_USE_EXP_116_12
    struct ExpValuesTableQ116_12
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_116_12
    #if TINYMIND_USE_EXP_117_11
    struct ExpValuesTableQ117_11
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_117_11
    #if TINYMIND_USE_EXP_118_10
    struct ExpValuesTableQ118_10
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_118_10
    #if TINYMIND_USE_EXP_119_9
    struct ExpValuesTableQ119_9
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_119_9
    #if TINYMIND_USE_EXP_120_8
    struct ExpValuesTableQ120_8
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_120_8
    #if TINYMIND_USE_EXP_121_7
    struct ExpValuesTableQ121_7
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_121_7
    #if TINYMIND_USE_EXP_122_6
    struct ExpValuesTableQ122_6
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_122_6
    #if TINYMIND_USE_EXP_123_5
    struct ExpValuesTableQ123_5
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_123_5
    #if TINYMIND_USE_EXP_124_4
    struct ExpValuesTableQ124_4
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_124_4
    #if TINYMIND_USE_EXP_125_3
    struct ExpValuesTableQ125_3
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_125_3
    #if TINYMIND_USE_EXP_126_2
    struct ExpValuesTableQ126_2
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_126_2
    #if TINYMIND_USE_EXP_127_1
    struct ExpValuesTableQ127_1
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_EXP_127_1
}
