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
    #if TINYMIND_USE_COS_1_127
    struct CosValuesTableQ1_127
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_1_127
    #if TINYMIND_USE_COS_2_126
    struct CosValuesTableQ2_126
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_2_126
    #if TINYMIND_USE_COS_3_125
    struct CosValuesTableQ3_125
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_3_125
    #if TINYMIND_USE_COS_4_124
    struct CosValuesTableQ4_124
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_4_124
    #if TINYMIND_USE_COS_5_123
    struct CosValuesTableQ5_123
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_5_123
    #if TINYMIND_USE_COS_6_122
    struct CosValuesTableQ6_122
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_6_122
    #if TINYMIND_USE_COS_7_121
    struct CosValuesTableQ7_121
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_7_121
    #if TINYMIND_USE_COS_8_120
    struct CosValuesTableQ8_120
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_8_120
    #if TINYMIND_USE_COS_9_119
    struct CosValuesTableQ9_119
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_9_119
    #if TINYMIND_USE_COS_10_118
    struct CosValuesTableQ10_118
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_10_118
    #if TINYMIND_USE_COS_11_117
    struct CosValuesTableQ11_117
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_11_117
    #if TINYMIND_USE_COS_12_116
    struct CosValuesTableQ12_116
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_12_116
    #if TINYMIND_USE_COS_13_115
    struct CosValuesTableQ13_115
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_13_115
    #if TINYMIND_USE_COS_14_114
    struct CosValuesTableQ14_114
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_14_114
    #if TINYMIND_USE_COS_15_113
    struct CosValuesTableQ15_113
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_15_113
    #if TINYMIND_USE_COS_16_112
    struct CosValuesTableQ16_112
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_16_112
    #if TINYMIND_USE_COS_17_111
    struct CosValuesTableQ17_111
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_17_111
    #if TINYMIND_USE_COS_18_110
    struct CosValuesTableQ18_110
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_18_110
    #if TINYMIND_USE_COS_19_109
    struct CosValuesTableQ19_109
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_19_109
    #if TINYMIND_USE_COS_20_108
    struct CosValuesTableQ20_108
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_20_108
    #if TINYMIND_USE_COS_21_107
    struct CosValuesTableQ21_107
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_21_107
    #if TINYMIND_USE_COS_22_106
    struct CosValuesTableQ22_106
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_22_106
    #if TINYMIND_USE_COS_23_105
    struct CosValuesTableQ23_105
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_23_105
    #if TINYMIND_USE_COS_24_104
    struct CosValuesTableQ24_104
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_24_104
    #if TINYMIND_USE_COS_25_103
    struct CosValuesTableQ25_103
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_25_103
    #if TINYMIND_USE_COS_26_102
    struct CosValuesTableQ26_102
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_26_102
    #if TINYMIND_USE_COS_27_101
    struct CosValuesTableQ27_101
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_27_101
    #if TINYMIND_USE_COS_28_100
    struct CosValuesTableQ28_100
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_28_100
    #if TINYMIND_USE_COS_29_99
    struct CosValuesTableQ29_99
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_29_99
    #if TINYMIND_USE_COS_30_98
    struct CosValuesTableQ30_98
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_30_98
    #if TINYMIND_USE_COS_31_97
    struct CosValuesTableQ31_97
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_31_97
    #if TINYMIND_USE_COS_32_96
    struct CosValuesTableQ32_96
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_32_96
    #if TINYMIND_USE_COS_33_95
    struct CosValuesTableQ33_95
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_33_95
    #if TINYMIND_USE_COS_34_94
    struct CosValuesTableQ34_94
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_34_94
    #if TINYMIND_USE_COS_35_93
    struct CosValuesTableQ35_93
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_35_93
    #if TINYMIND_USE_COS_36_92
    struct CosValuesTableQ36_92
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_36_92
    #if TINYMIND_USE_COS_37_91
    struct CosValuesTableQ37_91
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_37_91
    #if TINYMIND_USE_COS_38_90
    struct CosValuesTableQ38_90
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_38_90
    #if TINYMIND_USE_COS_39_89
    struct CosValuesTableQ39_89
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_39_89
    #if TINYMIND_USE_COS_40_88
    struct CosValuesTableQ40_88
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_40_88
    #if TINYMIND_USE_COS_41_87
    struct CosValuesTableQ41_87
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_41_87
    #if TINYMIND_USE_COS_42_86
    struct CosValuesTableQ42_86
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_42_86
    #if TINYMIND_USE_COS_43_85
    struct CosValuesTableQ43_85
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_43_85
    #if TINYMIND_USE_COS_44_84
    struct CosValuesTableQ44_84
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_44_84
    #if TINYMIND_USE_COS_45_83
    struct CosValuesTableQ45_83
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_45_83
    #if TINYMIND_USE_COS_46_82
    struct CosValuesTableQ46_82
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_46_82
    #if TINYMIND_USE_COS_47_81
    struct CosValuesTableQ47_81
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_47_81
    #if TINYMIND_USE_COS_48_80
    struct CosValuesTableQ48_80
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_48_80
    #if TINYMIND_USE_COS_49_79
    struct CosValuesTableQ49_79
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_49_79
    #if TINYMIND_USE_COS_50_78
    struct CosValuesTableQ50_78
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_50_78
    #if TINYMIND_USE_COS_51_77
    struct CosValuesTableQ51_77
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_51_77
    #if TINYMIND_USE_COS_52_76
    struct CosValuesTableQ52_76
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_52_76
    #if TINYMIND_USE_COS_53_75
    struct CosValuesTableQ53_75
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_53_75
    #if TINYMIND_USE_COS_54_74
    struct CosValuesTableQ54_74
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_54_74
    #if TINYMIND_USE_COS_55_73
    struct CosValuesTableQ55_73
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_55_73
    #if TINYMIND_USE_COS_56_72
    struct CosValuesTableQ56_72
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_56_72
    #if TINYMIND_USE_COS_57_71
    struct CosValuesTableQ57_71
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_57_71
    #if TINYMIND_USE_COS_58_70
    struct CosValuesTableQ58_70
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_58_70
    #if TINYMIND_USE_COS_59_69
    struct CosValuesTableQ59_69
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_59_69
    #if TINYMIND_USE_COS_60_68
    struct CosValuesTableQ60_68
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_60_68
    #if TINYMIND_USE_COS_61_67
    struct CosValuesTableQ61_67
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_61_67
    #if TINYMIND_USE_COS_62_66
    struct CosValuesTableQ62_66
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_62_66
    #if TINYMIND_USE_COS_63_65
    struct CosValuesTableQ63_65
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_63_65
    #if TINYMIND_USE_COS_64_64
    struct CosValuesTableQ64_64
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_64_64
    #if TINYMIND_USE_COS_65_63
    struct CosValuesTableQ65_63
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_65_63
    #if TINYMIND_USE_COS_66_62
    struct CosValuesTableQ66_62
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_66_62
    #if TINYMIND_USE_COS_67_61
    struct CosValuesTableQ67_61
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_67_61
    #if TINYMIND_USE_COS_68_60
    struct CosValuesTableQ68_60
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_68_60
    #if TINYMIND_USE_COS_69_59
    struct CosValuesTableQ69_59
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_69_59
    #if TINYMIND_USE_COS_70_58
    struct CosValuesTableQ70_58
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_70_58
    #if TINYMIND_USE_COS_71_57
    struct CosValuesTableQ71_57
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_71_57
    #if TINYMIND_USE_COS_72_56
    struct CosValuesTableQ72_56
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_72_56
    #if TINYMIND_USE_COS_73_55
    struct CosValuesTableQ73_55
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_73_55
    #if TINYMIND_USE_COS_74_54
    struct CosValuesTableQ74_54
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_74_54
    #if TINYMIND_USE_COS_75_53
    struct CosValuesTableQ75_53
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_75_53
    #if TINYMIND_USE_COS_76_52
    struct CosValuesTableQ76_52
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_76_52
    #if TINYMIND_USE_COS_77_51
    struct CosValuesTableQ77_51
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_77_51
    #if TINYMIND_USE_COS_78_50
    struct CosValuesTableQ78_50
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_78_50
    #if TINYMIND_USE_COS_79_49
    struct CosValuesTableQ79_49
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_79_49
    #if TINYMIND_USE_COS_80_48
    struct CosValuesTableQ80_48
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_80_48
    #if TINYMIND_USE_COS_81_47
    struct CosValuesTableQ81_47
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_81_47
    #if TINYMIND_USE_COS_82_46
    struct CosValuesTableQ82_46
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_82_46
    #if TINYMIND_USE_COS_83_45
    struct CosValuesTableQ83_45
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_83_45
    #if TINYMIND_USE_COS_84_44
    struct CosValuesTableQ84_44
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_84_44
    #if TINYMIND_USE_COS_85_43
    struct CosValuesTableQ85_43
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_85_43
    #if TINYMIND_USE_COS_86_42
    struct CosValuesTableQ86_42
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_86_42
    #if TINYMIND_USE_COS_87_41
    struct CosValuesTableQ87_41
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_87_41
    #if TINYMIND_USE_COS_88_40
    struct CosValuesTableQ88_40
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_88_40
    #if TINYMIND_USE_COS_89_39
    struct CosValuesTableQ89_39
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_89_39
    #if TINYMIND_USE_COS_90_38
    struct CosValuesTableQ90_38
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_90_38
    #if TINYMIND_USE_COS_91_37
    struct CosValuesTableQ91_37
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_91_37
    #if TINYMIND_USE_COS_92_36
    struct CosValuesTableQ92_36
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_92_36
    #if TINYMIND_USE_COS_93_35
    struct CosValuesTableQ93_35
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_93_35
    #if TINYMIND_USE_COS_94_34
    struct CosValuesTableQ94_34
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_94_34
    #if TINYMIND_USE_COS_95_33
    struct CosValuesTableQ95_33
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_95_33
    #if TINYMIND_USE_COS_96_32
    struct CosValuesTableQ96_32
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_96_32
    #if TINYMIND_USE_COS_97_31
    struct CosValuesTableQ97_31
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_97_31
    #if TINYMIND_USE_COS_98_30
    struct CosValuesTableQ98_30
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_98_30
    #if TINYMIND_USE_COS_99_29
    struct CosValuesTableQ99_29
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_99_29
    #if TINYMIND_USE_COS_100_28
    struct CosValuesTableQ100_28
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_100_28
    #if TINYMIND_USE_COS_101_27
    struct CosValuesTableQ101_27
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_101_27
    #if TINYMIND_USE_COS_102_26
    struct CosValuesTableQ102_26
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_102_26
    #if TINYMIND_USE_COS_103_25
    struct CosValuesTableQ103_25
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_103_25
    #if TINYMIND_USE_COS_104_24
    struct CosValuesTableQ104_24
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_104_24
    #if TINYMIND_USE_COS_105_23
    struct CosValuesTableQ105_23
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_105_23
    #if TINYMIND_USE_COS_106_22
    struct CosValuesTableQ106_22
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_106_22
    #if TINYMIND_USE_COS_107_21
    struct CosValuesTableQ107_21
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_107_21
    #if TINYMIND_USE_COS_108_20
    struct CosValuesTableQ108_20
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_108_20
    #if TINYMIND_USE_COS_109_19
    struct CosValuesTableQ109_19
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_109_19
    #if TINYMIND_USE_COS_110_18
    struct CosValuesTableQ110_18
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_110_18
    #if TINYMIND_USE_COS_111_17
    struct CosValuesTableQ111_17
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_111_17
    #if TINYMIND_USE_COS_112_16
    struct CosValuesTableQ112_16
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_112_16
    #if TINYMIND_USE_COS_113_15
    struct CosValuesTableQ113_15
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_113_15
    #if TINYMIND_USE_COS_114_14
    struct CosValuesTableQ114_14
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_114_14
    #if TINYMIND_USE_COS_115_13
    struct CosValuesTableQ115_13
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_115_13
    #if TINYMIND_USE_COS_116_12
    struct CosValuesTableQ116_12
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_116_12
    #if TINYMIND_USE_COS_117_11
    struct CosValuesTableQ117_11
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_117_11
    #if TINYMIND_USE_COS_118_10
    struct CosValuesTableQ118_10
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_118_10
    #if TINYMIND_USE_COS_119_9
    struct CosValuesTableQ119_9
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_119_9
    #if TINYMIND_USE_COS_120_8
    struct CosValuesTableQ120_8
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_120_8
    #if TINYMIND_USE_COS_121_7
    struct CosValuesTableQ121_7
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_121_7
    #if TINYMIND_USE_COS_122_6
    struct CosValuesTableQ122_6
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_122_6
    #if TINYMIND_USE_COS_123_5
    struct CosValuesTableQ123_5
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_123_5
    #if TINYMIND_USE_COS_124_4
    struct CosValuesTableQ124_4
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_124_4
    #if TINYMIND_USE_COS_125_3
    struct CosValuesTableQ125_3
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_125_3
    #if TINYMIND_USE_COS_126_2
    struct CosValuesTableQ126_2
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_126_2
    #if TINYMIND_USE_COS_127_1
    struct CosValuesTableQ127_1
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_127_1
}
