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
    #if TINYMIND_USE_SIGMOID_1_127
    struct SigmoidValuesTableQ1_127
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_1_127
    #if TINYMIND_USE_SIGMOID_2_126
    struct SigmoidValuesTableQ2_126
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_2_126
    #if TINYMIND_USE_SIGMOID_3_125
    struct SigmoidValuesTableQ3_125
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_3_125
    #if TINYMIND_USE_SIGMOID_4_124
    struct SigmoidValuesTableQ4_124
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_4_124
    #if TINYMIND_USE_SIGMOID_5_123
    struct SigmoidValuesTableQ5_123
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_5_123
    #if TINYMIND_USE_SIGMOID_6_122
    struct SigmoidValuesTableQ6_122
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_6_122
    #if TINYMIND_USE_SIGMOID_7_121
    struct SigmoidValuesTableQ7_121
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_7_121
    #if TINYMIND_USE_SIGMOID_8_120
    struct SigmoidValuesTableQ8_120
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_8_120
    #if TINYMIND_USE_SIGMOID_9_119
    struct SigmoidValuesTableQ9_119
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_9_119
    #if TINYMIND_USE_SIGMOID_10_118
    struct SigmoidValuesTableQ10_118
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_10_118
    #if TINYMIND_USE_SIGMOID_11_117
    struct SigmoidValuesTableQ11_117
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_11_117
    #if TINYMIND_USE_SIGMOID_12_116
    struct SigmoidValuesTableQ12_116
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_12_116
    #if TINYMIND_USE_SIGMOID_13_115
    struct SigmoidValuesTableQ13_115
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_13_115
    #if TINYMIND_USE_SIGMOID_14_114
    struct SigmoidValuesTableQ14_114
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_14_114
    #if TINYMIND_USE_SIGMOID_15_113
    struct SigmoidValuesTableQ15_113
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_15_113
    #if TINYMIND_USE_SIGMOID_16_112
    struct SigmoidValuesTableQ16_112
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_16_112
    #if TINYMIND_USE_SIGMOID_17_111
    struct SigmoidValuesTableQ17_111
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_17_111
    #if TINYMIND_USE_SIGMOID_18_110
    struct SigmoidValuesTableQ18_110
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_18_110
    #if TINYMIND_USE_SIGMOID_19_109
    struct SigmoidValuesTableQ19_109
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_19_109
    #if TINYMIND_USE_SIGMOID_20_108
    struct SigmoidValuesTableQ20_108
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_20_108
    #if TINYMIND_USE_SIGMOID_21_107
    struct SigmoidValuesTableQ21_107
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_21_107
    #if TINYMIND_USE_SIGMOID_22_106
    struct SigmoidValuesTableQ22_106
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_22_106
    #if TINYMIND_USE_SIGMOID_23_105
    struct SigmoidValuesTableQ23_105
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_23_105
    #if TINYMIND_USE_SIGMOID_24_104
    struct SigmoidValuesTableQ24_104
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_24_104
    #if TINYMIND_USE_SIGMOID_25_103
    struct SigmoidValuesTableQ25_103
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_25_103
    #if TINYMIND_USE_SIGMOID_26_102
    struct SigmoidValuesTableQ26_102
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_26_102
    #if TINYMIND_USE_SIGMOID_27_101
    struct SigmoidValuesTableQ27_101
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_27_101
    #if TINYMIND_USE_SIGMOID_28_100
    struct SigmoidValuesTableQ28_100
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_28_100
    #if TINYMIND_USE_SIGMOID_29_99
    struct SigmoidValuesTableQ29_99
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_29_99
    #if TINYMIND_USE_SIGMOID_30_98
    struct SigmoidValuesTableQ30_98
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_30_98
    #if TINYMIND_USE_SIGMOID_31_97
    struct SigmoidValuesTableQ31_97
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_31_97
    #if TINYMIND_USE_SIGMOID_32_96
    struct SigmoidValuesTableQ32_96
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_32_96
    #if TINYMIND_USE_SIGMOID_33_95
    struct SigmoidValuesTableQ33_95
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_33_95
    #if TINYMIND_USE_SIGMOID_34_94
    struct SigmoidValuesTableQ34_94
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_34_94
    #if TINYMIND_USE_SIGMOID_35_93
    struct SigmoidValuesTableQ35_93
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_35_93
    #if TINYMIND_USE_SIGMOID_36_92
    struct SigmoidValuesTableQ36_92
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_36_92
    #if TINYMIND_USE_SIGMOID_37_91
    struct SigmoidValuesTableQ37_91
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_37_91
    #if TINYMIND_USE_SIGMOID_38_90
    struct SigmoidValuesTableQ38_90
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_38_90
    #if TINYMIND_USE_SIGMOID_39_89
    struct SigmoidValuesTableQ39_89
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_39_89
    #if TINYMIND_USE_SIGMOID_40_88
    struct SigmoidValuesTableQ40_88
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_40_88
    #if TINYMIND_USE_SIGMOID_41_87
    struct SigmoidValuesTableQ41_87
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_41_87
    #if TINYMIND_USE_SIGMOID_42_86
    struct SigmoidValuesTableQ42_86
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_42_86
    #if TINYMIND_USE_SIGMOID_43_85
    struct SigmoidValuesTableQ43_85
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_43_85
    #if TINYMIND_USE_SIGMOID_44_84
    struct SigmoidValuesTableQ44_84
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_44_84
    #if TINYMIND_USE_SIGMOID_45_83
    struct SigmoidValuesTableQ45_83
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_45_83
    #if TINYMIND_USE_SIGMOID_46_82
    struct SigmoidValuesTableQ46_82
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_46_82
    #if TINYMIND_USE_SIGMOID_47_81
    struct SigmoidValuesTableQ47_81
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_47_81
    #if TINYMIND_USE_SIGMOID_48_80
    struct SigmoidValuesTableQ48_80
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_48_80
    #if TINYMIND_USE_SIGMOID_49_79
    struct SigmoidValuesTableQ49_79
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_49_79
    #if TINYMIND_USE_SIGMOID_50_78
    struct SigmoidValuesTableQ50_78
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_50_78
    #if TINYMIND_USE_SIGMOID_51_77
    struct SigmoidValuesTableQ51_77
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_51_77
    #if TINYMIND_USE_SIGMOID_52_76
    struct SigmoidValuesTableQ52_76
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_52_76
    #if TINYMIND_USE_SIGMOID_53_75
    struct SigmoidValuesTableQ53_75
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_53_75
    #if TINYMIND_USE_SIGMOID_54_74
    struct SigmoidValuesTableQ54_74
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_54_74
    #if TINYMIND_USE_SIGMOID_55_73
    struct SigmoidValuesTableQ55_73
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_55_73
    #if TINYMIND_USE_SIGMOID_56_72
    struct SigmoidValuesTableQ56_72
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_56_72
    #if TINYMIND_USE_SIGMOID_57_71
    struct SigmoidValuesTableQ57_71
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_57_71
    #if TINYMIND_USE_SIGMOID_58_70
    struct SigmoidValuesTableQ58_70
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_58_70
    #if TINYMIND_USE_SIGMOID_59_69
    struct SigmoidValuesTableQ59_69
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_59_69
    #if TINYMIND_USE_SIGMOID_60_68
    struct SigmoidValuesTableQ60_68
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_60_68
    #if TINYMIND_USE_SIGMOID_61_67
    struct SigmoidValuesTableQ61_67
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_61_67
    #if TINYMIND_USE_SIGMOID_62_66
    struct SigmoidValuesTableQ62_66
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_62_66
    #if TINYMIND_USE_SIGMOID_63_65
    struct SigmoidValuesTableQ63_65
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_63_65
    #if TINYMIND_USE_SIGMOID_64_64
    struct SigmoidValuesTableQ64_64
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_64_64
    #if TINYMIND_USE_SIGMOID_65_63
    struct SigmoidValuesTableQ65_63
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_65_63
    #if TINYMIND_USE_SIGMOID_66_62
    struct SigmoidValuesTableQ66_62
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_66_62
    #if TINYMIND_USE_SIGMOID_67_61
    struct SigmoidValuesTableQ67_61
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_67_61
    #if TINYMIND_USE_SIGMOID_68_60
    struct SigmoidValuesTableQ68_60
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_68_60
    #if TINYMIND_USE_SIGMOID_69_59
    struct SigmoidValuesTableQ69_59
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_69_59
    #if TINYMIND_USE_SIGMOID_70_58
    struct SigmoidValuesTableQ70_58
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_70_58
    #if TINYMIND_USE_SIGMOID_71_57
    struct SigmoidValuesTableQ71_57
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_71_57
    #if TINYMIND_USE_SIGMOID_72_56
    struct SigmoidValuesTableQ72_56
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_72_56
    #if TINYMIND_USE_SIGMOID_73_55
    struct SigmoidValuesTableQ73_55
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_73_55
    #if TINYMIND_USE_SIGMOID_74_54
    struct SigmoidValuesTableQ74_54
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_74_54
    #if TINYMIND_USE_SIGMOID_75_53
    struct SigmoidValuesTableQ75_53
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_75_53
    #if TINYMIND_USE_SIGMOID_76_52
    struct SigmoidValuesTableQ76_52
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_76_52
    #if TINYMIND_USE_SIGMOID_77_51
    struct SigmoidValuesTableQ77_51
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_77_51
    #if TINYMIND_USE_SIGMOID_78_50
    struct SigmoidValuesTableQ78_50
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_78_50
    #if TINYMIND_USE_SIGMOID_79_49
    struct SigmoidValuesTableQ79_49
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_79_49
    #if TINYMIND_USE_SIGMOID_80_48
    struct SigmoidValuesTableQ80_48
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_80_48
    #if TINYMIND_USE_SIGMOID_81_47
    struct SigmoidValuesTableQ81_47
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_81_47
    #if TINYMIND_USE_SIGMOID_82_46
    struct SigmoidValuesTableQ82_46
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_82_46
    #if TINYMIND_USE_SIGMOID_83_45
    struct SigmoidValuesTableQ83_45
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_83_45
    #if TINYMIND_USE_SIGMOID_84_44
    struct SigmoidValuesTableQ84_44
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_84_44
    #if TINYMIND_USE_SIGMOID_85_43
    struct SigmoidValuesTableQ85_43
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_85_43
    #if TINYMIND_USE_SIGMOID_86_42
    struct SigmoidValuesTableQ86_42
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_86_42
    #if TINYMIND_USE_SIGMOID_87_41
    struct SigmoidValuesTableQ87_41
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_87_41
    #if TINYMIND_USE_SIGMOID_88_40
    struct SigmoidValuesTableQ88_40
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_88_40
    #if TINYMIND_USE_SIGMOID_89_39
    struct SigmoidValuesTableQ89_39
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_89_39
    #if TINYMIND_USE_SIGMOID_90_38
    struct SigmoidValuesTableQ90_38
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_90_38
    #if TINYMIND_USE_SIGMOID_91_37
    struct SigmoidValuesTableQ91_37
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_91_37
    #if TINYMIND_USE_SIGMOID_92_36
    struct SigmoidValuesTableQ92_36
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_92_36
    #if TINYMIND_USE_SIGMOID_93_35
    struct SigmoidValuesTableQ93_35
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_93_35
    #if TINYMIND_USE_SIGMOID_94_34
    struct SigmoidValuesTableQ94_34
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_94_34
    #if TINYMIND_USE_SIGMOID_95_33
    struct SigmoidValuesTableQ95_33
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_95_33
    #if TINYMIND_USE_SIGMOID_96_32
    struct SigmoidValuesTableQ96_32
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_96_32
    #if TINYMIND_USE_SIGMOID_97_31
    struct SigmoidValuesTableQ97_31
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_97_31
    #if TINYMIND_USE_SIGMOID_98_30
    struct SigmoidValuesTableQ98_30
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_98_30
    #if TINYMIND_USE_SIGMOID_99_29
    struct SigmoidValuesTableQ99_29
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_99_29
    #if TINYMIND_USE_SIGMOID_100_28
    struct SigmoidValuesTableQ100_28
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_100_28
    #if TINYMIND_USE_SIGMOID_101_27
    struct SigmoidValuesTableQ101_27
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_101_27
    #if TINYMIND_USE_SIGMOID_102_26
    struct SigmoidValuesTableQ102_26
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_102_26
    #if TINYMIND_USE_SIGMOID_103_25
    struct SigmoidValuesTableQ103_25
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_103_25
    #if TINYMIND_USE_SIGMOID_104_24
    struct SigmoidValuesTableQ104_24
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_104_24
    #if TINYMIND_USE_SIGMOID_105_23
    struct SigmoidValuesTableQ105_23
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_105_23
    #if TINYMIND_USE_SIGMOID_106_22
    struct SigmoidValuesTableQ106_22
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_106_22
    #if TINYMIND_USE_SIGMOID_107_21
    struct SigmoidValuesTableQ107_21
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_107_21
    #if TINYMIND_USE_SIGMOID_108_20
    struct SigmoidValuesTableQ108_20
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_108_20
    #if TINYMIND_USE_SIGMOID_109_19
    struct SigmoidValuesTableQ109_19
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_109_19
    #if TINYMIND_USE_SIGMOID_110_18
    struct SigmoidValuesTableQ110_18
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_110_18
    #if TINYMIND_USE_SIGMOID_111_17
    struct SigmoidValuesTableQ111_17
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_111_17
    #if TINYMIND_USE_SIGMOID_112_16
    struct SigmoidValuesTableQ112_16
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_112_16
    #if TINYMIND_USE_SIGMOID_113_15
    struct SigmoidValuesTableQ113_15
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_113_15
    #if TINYMIND_USE_SIGMOID_114_14
    struct SigmoidValuesTableQ114_14
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_114_14
    #if TINYMIND_USE_SIGMOID_115_13
    struct SigmoidValuesTableQ115_13
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_115_13
    #if TINYMIND_USE_SIGMOID_116_12
    struct SigmoidValuesTableQ116_12
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_116_12
    #if TINYMIND_USE_SIGMOID_117_11
    struct SigmoidValuesTableQ117_11
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_117_11
    #if TINYMIND_USE_SIGMOID_118_10
    struct SigmoidValuesTableQ118_10
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_118_10
    #if TINYMIND_USE_SIGMOID_119_9
    struct SigmoidValuesTableQ119_9
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_119_9
    #if TINYMIND_USE_SIGMOID_120_8
    struct SigmoidValuesTableQ120_8
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_120_8
    #if TINYMIND_USE_SIGMOID_121_7
    struct SigmoidValuesTableQ121_7
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_121_7
    #if TINYMIND_USE_SIGMOID_122_6
    struct SigmoidValuesTableQ122_6
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_122_6
    #if TINYMIND_USE_SIGMOID_123_5
    struct SigmoidValuesTableQ123_5
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_123_5
    #if TINYMIND_USE_SIGMOID_124_4
    struct SigmoidValuesTableQ124_4
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_124_4
    #if TINYMIND_USE_SIGMOID_125_3
    struct SigmoidValuesTableQ125_3
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_125_3
    #if TINYMIND_USE_SIGMOID_126_2
    struct SigmoidValuesTableQ126_2
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_126_2
    #if TINYMIND_USE_SIGMOID_127_1
    struct SigmoidValuesTableQ127_1
    {
        static const uint128_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIGMOID_127_1
}
