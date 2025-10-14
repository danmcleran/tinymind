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
    #if TINYMIND_USE_TANH_1_127
    struct TanhValuesTableQ1_127
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_1_127
    #if TINYMIND_USE_TANH_2_126
    struct TanhValuesTableQ2_126
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_2_126
    #if TINYMIND_USE_TANH_3_125
    struct TanhValuesTableQ3_125
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_3_125
    #if TINYMIND_USE_TANH_4_124
    struct TanhValuesTableQ4_124
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_4_124
    #if TINYMIND_USE_TANH_5_123
    struct TanhValuesTableQ5_123
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_5_123
    #if TINYMIND_USE_TANH_6_122
    struct TanhValuesTableQ6_122
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_6_122
    #if TINYMIND_USE_TANH_7_121
    struct TanhValuesTableQ7_121
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_7_121
    #if TINYMIND_USE_TANH_8_120
    struct TanhValuesTableQ8_120
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_8_120
    #if TINYMIND_USE_TANH_9_119
    struct TanhValuesTableQ9_119
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_9_119
    #if TINYMIND_USE_TANH_10_118
    struct TanhValuesTableQ10_118
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_10_118
    #if TINYMIND_USE_TANH_11_117
    struct TanhValuesTableQ11_117
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_11_117
    #if TINYMIND_USE_TANH_12_116
    struct TanhValuesTableQ12_116
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_12_116
    #if TINYMIND_USE_TANH_13_115
    struct TanhValuesTableQ13_115
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_13_115
    #if TINYMIND_USE_TANH_14_114
    struct TanhValuesTableQ14_114
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_14_114
    #if TINYMIND_USE_TANH_15_113
    struct TanhValuesTableQ15_113
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_15_113
    #if TINYMIND_USE_TANH_16_112
    struct TanhValuesTableQ16_112
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_16_112
    #if TINYMIND_USE_TANH_17_111
    struct TanhValuesTableQ17_111
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_17_111
    #if TINYMIND_USE_TANH_18_110
    struct TanhValuesTableQ18_110
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_18_110
    #if TINYMIND_USE_TANH_19_109
    struct TanhValuesTableQ19_109
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_19_109
    #if TINYMIND_USE_TANH_20_108
    struct TanhValuesTableQ20_108
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_20_108
    #if TINYMIND_USE_TANH_21_107
    struct TanhValuesTableQ21_107
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_21_107
    #if TINYMIND_USE_TANH_22_106
    struct TanhValuesTableQ22_106
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_22_106
    #if TINYMIND_USE_TANH_23_105
    struct TanhValuesTableQ23_105
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_23_105
    #if TINYMIND_USE_TANH_24_104
    struct TanhValuesTableQ24_104
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_24_104
    #if TINYMIND_USE_TANH_25_103
    struct TanhValuesTableQ25_103
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_25_103
    #if TINYMIND_USE_TANH_26_102
    struct TanhValuesTableQ26_102
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_26_102
    #if TINYMIND_USE_TANH_27_101
    struct TanhValuesTableQ27_101
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_27_101
    #if TINYMIND_USE_TANH_28_100
    struct TanhValuesTableQ28_100
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_28_100
    #if TINYMIND_USE_TANH_29_99
    struct TanhValuesTableQ29_99
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_29_99
    #if TINYMIND_USE_TANH_30_98
    struct TanhValuesTableQ30_98
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_30_98
    #if TINYMIND_USE_TANH_31_97
    struct TanhValuesTableQ31_97
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_31_97
    #if TINYMIND_USE_TANH_32_96
    struct TanhValuesTableQ32_96
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_32_96
    #if TINYMIND_USE_TANH_33_95
    struct TanhValuesTableQ33_95
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_33_95
    #if TINYMIND_USE_TANH_34_94
    struct TanhValuesTableQ34_94
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_34_94
    #if TINYMIND_USE_TANH_35_93
    struct TanhValuesTableQ35_93
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_35_93
    #if TINYMIND_USE_TANH_36_92
    struct TanhValuesTableQ36_92
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_36_92
    #if TINYMIND_USE_TANH_37_91
    struct TanhValuesTableQ37_91
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_37_91
    #if TINYMIND_USE_TANH_38_90
    struct TanhValuesTableQ38_90
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_38_90
    #if TINYMIND_USE_TANH_39_89
    struct TanhValuesTableQ39_89
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_39_89
    #if TINYMIND_USE_TANH_40_88
    struct TanhValuesTableQ40_88
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_40_88
    #if TINYMIND_USE_TANH_41_87
    struct TanhValuesTableQ41_87
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_41_87
    #if TINYMIND_USE_TANH_42_86
    struct TanhValuesTableQ42_86
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_42_86
    #if TINYMIND_USE_TANH_43_85
    struct TanhValuesTableQ43_85
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_43_85
    #if TINYMIND_USE_TANH_44_84
    struct TanhValuesTableQ44_84
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_44_84
    #if TINYMIND_USE_TANH_45_83
    struct TanhValuesTableQ45_83
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_45_83
    #if TINYMIND_USE_TANH_46_82
    struct TanhValuesTableQ46_82
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_46_82
    #if TINYMIND_USE_TANH_47_81
    struct TanhValuesTableQ47_81
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_47_81
    #if TINYMIND_USE_TANH_48_80
    struct TanhValuesTableQ48_80
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_48_80
    #if TINYMIND_USE_TANH_49_79
    struct TanhValuesTableQ49_79
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_49_79
    #if TINYMIND_USE_TANH_50_78
    struct TanhValuesTableQ50_78
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_50_78
    #if TINYMIND_USE_TANH_51_77
    struct TanhValuesTableQ51_77
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_51_77
    #if TINYMIND_USE_TANH_52_76
    struct TanhValuesTableQ52_76
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_52_76
    #if TINYMIND_USE_TANH_53_75
    struct TanhValuesTableQ53_75
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_53_75
    #if TINYMIND_USE_TANH_54_74
    struct TanhValuesTableQ54_74
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_54_74
    #if TINYMIND_USE_TANH_55_73
    struct TanhValuesTableQ55_73
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_55_73
    #if TINYMIND_USE_TANH_56_72
    struct TanhValuesTableQ56_72
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_56_72
    #if TINYMIND_USE_TANH_57_71
    struct TanhValuesTableQ57_71
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_57_71
    #if TINYMIND_USE_TANH_58_70
    struct TanhValuesTableQ58_70
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_58_70
    #if TINYMIND_USE_TANH_59_69
    struct TanhValuesTableQ59_69
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_59_69
    #if TINYMIND_USE_TANH_60_68
    struct TanhValuesTableQ60_68
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_60_68
    #if TINYMIND_USE_TANH_61_67
    struct TanhValuesTableQ61_67
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_61_67
    #if TINYMIND_USE_TANH_62_66
    struct TanhValuesTableQ62_66
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_62_66
    #if TINYMIND_USE_TANH_63_65
    struct TanhValuesTableQ63_65
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_63_65
    #if TINYMIND_USE_TANH_64_64
    struct TanhValuesTableQ64_64
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_64_64
    #if TINYMIND_USE_TANH_65_63
    struct TanhValuesTableQ65_63
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_65_63
    #if TINYMIND_USE_TANH_66_62
    struct TanhValuesTableQ66_62
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_66_62
    #if TINYMIND_USE_TANH_67_61
    struct TanhValuesTableQ67_61
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_67_61
    #if TINYMIND_USE_TANH_68_60
    struct TanhValuesTableQ68_60
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_68_60
    #if TINYMIND_USE_TANH_69_59
    struct TanhValuesTableQ69_59
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_69_59
    #if TINYMIND_USE_TANH_70_58
    struct TanhValuesTableQ70_58
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_70_58
    #if TINYMIND_USE_TANH_71_57
    struct TanhValuesTableQ71_57
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_71_57
    #if TINYMIND_USE_TANH_72_56
    struct TanhValuesTableQ72_56
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_72_56
    #if TINYMIND_USE_TANH_73_55
    struct TanhValuesTableQ73_55
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_73_55
    #if TINYMIND_USE_TANH_74_54
    struct TanhValuesTableQ74_54
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_74_54
    #if TINYMIND_USE_TANH_75_53
    struct TanhValuesTableQ75_53
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_75_53
    #if TINYMIND_USE_TANH_76_52
    struct TanhValuesTableQ76_52
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_76_52
    #if TINYMIND_USE_TANH_77_51
    struct TanhValuesTableQ77_51
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_77_51
    #if TINYMIND_USE_TANH_78_50
    struct TanhValuesTableQ78_50
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_78_50
    #if TINYMIND_USE_TANH_79_49
    struct TanhValuesTableQ79_49
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_79_49
    #if TINYMIND_USE_TANH_80_48
    struct TanhValuesTableQ80_48
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_80_48
    #if TINYMIND_USE_TANH_81_47
    struct TanhValuesTableQ81_47
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_81_47
    #if TINYMIND_USE_TANH_82_46
    struct TanhValuesTableQ82_46
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_82_46
    #if TINYMIND_USE_TANH_83_45
    struct TanhValuesTableQ83_45
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_83_45
    #if TINYMIND_USE_TANH_84_44
    struct TanhValuesTableQ84_44
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_84_44
    #if TINYMIND_USE_TANH_85_43
    struct TanhValuesTableQ85_43
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_85_43
    #if TINYMIND_USE_TANH_86_42
    struct TanhValuesTableQ86_42
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_86_42
    #if TINYMIND_USE_TANH_87_41
    struct TanhValuesTableQ87_41
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_87_41
    #if TINYMIND_USE_TANH_88_40
    struct TanhValuesTableQ88_40
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_88_40
    #if TINYMIND_USE_TANH_89_39
    struct TanhValuesTableQ89_39
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_89_39
    #if TINYMIND_USE_TANH_90_38
    struct TanhValuesTableQ90_38
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_90_38
    #if TINYMIND_USE_TANH_91_37
    struct TanhValuesTableQ91_37
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_91_37
    #if TINYMIND_USE_TANH_92_36
    struct TanhValuesTableQ92_36
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_92_36
    #if TINYMIND_USE_TANH_93_35
    struct TanhValuesTableQ93_35
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_93_35
    #if TINYMIND_USE_TANH_94_34
    struct TanhValuesTableQ94_34
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_94_34
    #if TINYMIND_USE_TANH_95_33
    struct TanhValuesTableQ95_33
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_95_33
    #if TINYMIND_USE_TANH_96_32
    struct TanhValuesTableQ96_32
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_96_32
    #if TINYMIND_USE_TANH_97_31
    struct TanhValuesTableQ97_31
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_97_31
    #if TINYMIND_USE_TANH_98_30
    struct TanhValuesTableQ98_30
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_98_30
    #if TINYMIND_USE_TANH_99_29
    struct TanhValuesTableQ99_29
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_99_29
    #if TINYMIND_USE_TANH_100_28
    struct TanhValuesTableQ100_28
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_100_28
    #if TINYMIND_USE_TANH_101_27
    struct TanhValuesTableQ101_27
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_101_27
    #if TINYMIND_USE_TANH_102_26
    struct TanhValuesTableQ102_26
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_102_26
    #if TINYMIND_USE_TANH_103_25
    struct TanhValuesTableQ103_25
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_103_25
    #if TINYMIND_USE_TANH_104_24
    struct TanhValuesTableQ104_24
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_104_24
    #if TINYMIND_USE_TANH_105_23
    struct TanhValuesTableQ105_23
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_105_23
    #if TINYMIND_USE_TANH_106_22
    struct TanhValuesTableQ106_22
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_106_22
    #if TINYMIND_USE_TANH_107_21
    struct TanhValuesTableQ107_21
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_107_21
    #if TINYMIND_USE_TANH_108_20
    struct TanhValuesTableQ108_20
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_108_20
    #if TINYMIND_USE_TANH_109_19
    struct TanhValuesTableQ109_19
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_109_19
    #if TINYMIND_USE_TANH_110_18
    struct TanhValuesTableQ110_18
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_110_18
    #if TINYMIND_USE_TANH_111_17
    struct TanhValuesTableQ111_17
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_111_17
    #if TINYMIND_USE_TANH_112_16
    struct TanhValuesTableQ112_16
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_112_16
    #if TINYMIND_USE_TANH_113_15
    struct TanhValuesTableQ113_15
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_113_15
    #if TINYMIND_USE_TANH_114_14
    struct TanhValuesTableQ114_14
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_114_14
    #if TINYMIND_USE_TANH_115_13
    struct TanhValuesTableQ115_13
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_115_13
    #if TINYMIND_USE_TANH_116_12
    struct TanhValuesTableQ116_12
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_116_12
    #if TINYMIND_USE_TANH_117_11
    struct TanhValuesTableQ117_11
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_117_11
    #if TINYMIND_USE_TANH_118_10
    struct TanhValuesTableQ118_10
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_118_10
    #if TINYMIND_USE_TANH_119_9
    struct TanhValuesTableQ119_9
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_119_9
    #if TINYMIND_USE_TANH_120_8
    struct TanhValuesTableQ120_8
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_120_8
    #if TINYMIND_USE_TANH_121_7
    struct TanhValuesTableQ121_7
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_121_7
    #if TINYMIND_USE_TANH_122_6
    struct TanhValuesTableQ122_6
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_122_6
    #if TINYMIND_USE_TANH_123_5
    struct TanhValuesTableQ123_5
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_123_5
    #if TINYMIND_USE_TANH_124_4
    struct TanhValuesTableQ124_4
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_124_4
    #if TINYMIND_USE_TANH_125_3
    struct TanhValuesTableQ125_3
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_125_3
    #if TINYMIND_USE_TANH_126_2
    struct TanhValuesTableQ126_2
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_126_2
    #if TINYMIND_USE_TANH_127_1
    struct TanhValuesTableQ127_1
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_127_1
}
