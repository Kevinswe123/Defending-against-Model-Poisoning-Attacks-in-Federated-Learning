
#xinzeng 

sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.005 50 0 1 2 no fltrust 0.1 2 0.2 False
sh run_test_byz_p1.sh -1 MNIST cnn 0.5 10 64 0.005 50 0 1 2 no fltrust 0.1 0 0.2 False

ceshi watermarkfl
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.005 50 100 1 2 trim_attack watermarkfl 0.1 2 0.2 False

updated:
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 5 64 0.2 100 100 110 2 trim_attack watermarkfl 0.1 2 0.2 False

success lr=0.001:
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.1 100 100 110 4 trim_attack watermarkfl 0.1 2 0.2 False

i.i.d.
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.1 10 64 0.00005 10 100 110(12-16) 4 trim_attack watermarkfl 0.1 2 0.2 False
non i.i.d.
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.00005 10 100 110(12-16) 4 trim_attack watermarkfl 0.1 2 0.2 False 0.01
non i.i.d. dbdataset=CIFAR100 localLR=0.0005
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.0001 50 100 1 4 trim_attack watermarkfl 0.1 2 0.2 False 0.01 CIFAR100
#result: test_acc_list [0.1339, 0.1492, 0.1524, 0.1648, 0.1728, 0.1772, 0.1844, 0.1828, 0.1877, 0.1931, 0.1949, 0.1967, 0.2013, 0.205, 0.2069, 0.209, 0.2105, 0.2163, 0.2172, 0.221]
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.0001 100 100 2 2 trim_attack watermarkfl 0.1 2 0.2 False 0.01 CIFAR100
#:test_acc_list [0.1366, 0.1365, 0.1387, 0.1468, 0.1621, 0.1761, 0.1786, 0.1874, 0.1941, 0.2013, 0.205, 0.2067, 0.2116, 0.2133, 0.2193, 0.2206, 0.2254, 0.2281, 0.2347, 0.2376, 0.2353, 0.2363, 0.2373, 0.2433, 0.2465, 0.2478, 0.2472, 0.2507, 0.2519, 0.252, 0.2525, 0.2541, 0.2543, 0.2571, 0.2586, 0.2579, 0.2583, 0.2562, 0.2577, 0.2586, 0.2602, 0.2594, 0.2622, 0.2614, 0.2648, 0.2672, 0.264, 0.2657, 0.2693, 0.2659, 0.2677, 0.265, 0.2701, 0.2706, 0.2748, 0.2722, 0.2737, 0.2744, 0.2727, 0.2722, 0.2765, 0.276, 0.2766, 0.2768, 0.2761, 0.2774, 0.2795, 0.2817, 0.2814, 0.2816, 0.2815, 0.28, 0.2807, 0.2829, 0.2792, 0.2817, 0.2806, 0.282, 0.2831, 0.2839, 0.2843, 0.2843, 0.285, 0.2858, 0.2861, 0.2868, 0.2878, 0.2877, 0.2879, 0.2885, 0.2893, 0.291, 0.2907, 0.2919, 0.2928, 0.2905, 0.2901, 0.2899, 0.2921, 0.2913]
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.001 100 100 2 2 trim_attack watermarkfl 0.1 2 0.2 False 0.01 CIFAR100
#:test_acc_list [0.1557, 0.1562, 0.1627, 0.1814, 0.1993, 0.2038, 0.2047, 0.2139, 0.2143, 0.2119, 0.2057, 0.2145, 0.223, 0.2226, 0.2192, 0.2208, 0.2237, 0.2287, 0.2271, 0.2311, 0.2335, 0.2362, 0.2373, 0.2398, 0.2413, 0.2411, 0.2412, 0.243, 0.2423, 0.2414, 0.2511, 0.2532, 0.2545, 0.2578, 0.2629, 0.2622, 0.2604, 0.2672, 0.2753, 0.2757, 0.2782, 0.2783, 0.2792, 0.2818, 0.285, 0.2873, 0.2889, 0.2906, 0.2904, 0.2971, 0.2915, 0.2932, 0.2956, 0.3009, 0.3018, 0.299, 0.3042, 0.3083, 0.3047, 0.3015, 0.3014, 0.305, 0.3052, 0.3111, 0.3109, 0.315, 0.3167, 0.3212, 0.3157, 0.3133, 0.3024, 0.3173, 0.3263, 0.3298, 0.3273, 0.3301, 0.3301, 0.3343, 0.3129, 0.3354, 0.3411, 0.3453, 0.3362, 0.3305, 0.3313, 0.3187, 0.3355, 0.3449, 0.3518, 0.3415, 0.3409, 0.3142, 0.3282, 0.3481, 0.343, 0.3338, 0.3543, 0.3498, 0.3519, 0.3551]
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.001 1000 100 2 2 trim_attack watermarkfl 0.1 2 0.2 False 0.01 CIFAR100
#:test_acc_list [0.2119, 0.2311, 0.2414, 0.2757, 0.2971, 0.3015, 0.3133, 0.3354, 0.3415, 0.3551, 0.3659, 0.3646, 0.3724, 0.3866, 0.3996, 0.3973, 0.3871, 0.3971, 0.4101, 0.4202, 0.4284, 0.4188, 0.4363, 0.4396, 0.4175, 0.4402, 0.4455, 0.4452, 0.449, 0.4462, 0.4504, 0.4457, 0.4469, 0.4594, 0.4583, 0.471, 0.464, 0.4694, 0.4633, 0.4686, 0.4718, 0.4677, 0.4729, 0.4762, 0.4746, 0.4778, 0.4726, 0.483, 0.4782, 0.4868, 0.478, 0.4867, 0.4791, 0.4865, 0.4823, 0.4853, 0.4804, 0.4869, 0.492, 0.4904, 0.4785, 0.4871, 0.4944, 0.4885, 0.4942, 0.4966, 0.4946, 0.5, 0.4874, 0.493, 0.5057, 0.5021, 0.5034, 0.4901, 0.4973, 0.5026, 0.5033, 0.5009, 0.5049, 0.4982, 0.5148, 0.4989, 0.5049, 0.5038, 0.4929, 0.5085, 0.5081, 0.5032, 0.5169, 0.5041, 0.5002, 0.5014, 0.5104, 0.5089, 0.4933, 0.52, 0.5109, 0.5173, 0.5135, 0.5145]
seed 176 works

########## threshold = 0.005, local lr and test 3rounds lr = 0.0001
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.001 1000 100 6 4 trim_attack watermarkfl 0.1 2 0.2 False 0.005 CIFAR100
##############FLTRUST############
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.001 1000 100 2 2 trim_attack fltrust 0.1 2 0.2 Skip 0.01 CIFAR100
#:test_acc_list [0.1, 0.1, 0.0999, 0.0999, 0.1, 0.1001, 0.1, 0.1017, 0.1004, 0.1012, 0.1036, 0.1074, 0.1063, 0.1068, 0.1135, 0.1271, 0.168, 0.192, 0.2078, 0.204, 0.2172, 0.2225, 0.2236, 0.2237, 0.2311, 0.2415, 0.2442, 0.2477, 0.2499, 0.2463, 0.2508, 0.2477, 0.2401, 0.2564, 0.2596, 0.2613, 0.2495, 0.2552, 0.2266, 0.2533, 0.2522, 0.2675, 0.2558, 0.2903, 0.2875, 0.2878, 0.2851, 0.3003, 0.2919, 0.3014, 0.2956, 0.2922, 0.3208, 0.2944, 0.3251, 0.3291, 0.3335, 0.3322, 0.3286, 0.3362, 0.3213, 0.3332, 0.3278, 0.3311, 0.3229, 0.3513, 0.3564, 0.3245, 0.323, 0.335, 0.3591, 0.3388, 0.3468, 0.3588, 0.3541, 0.3583, 0.3631, 0.3658, 0.3736, 0.3486, 0.3699, 0.3597, 0.3764, 0.3729, 0.3758, 0.3825, 0.382, 0.3649, 0.3759, 0.3812, 0.3768, 0.381, 0.3907, 0.3825, 0.389, 0.3717, 0.3637, 0.3866, 0.3885, 0.405]

############ FedAvg setting  ##################
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.005 1000 100 2 2 no fedavgfl 0.1 2 0.2 Skip 0.01 CIFAR100
#:test_acc_list [0.1219, 0.1238, 0.1638, 0.0992, 0.1003, 0.1659, 0.1541, 0.1467, 0.1, 0.1246, 0.2269, 0.2436, 0.102, 0.1079, 0.1335, 0.2093, 0.2306, 0.1877, 0.257, 0.2229, 0.2287, 0.2827, 0.2195, 0.109, 0.1915, 0.2351, 0.2754, 0.2723, 0.2254, 0.269, 0.1002, 0.102, 0.2641, 0.2895, 0.3062, 0.2513, 0.269, 0.2602, 0.2868, 0.1398, 0.3184, 0.3436, 0.3576, 0.3435, 0.3583, 0.3502, 0.3605, 0.3401, 0.3743, 0.3351, 0.3913, 0.382, 0.359, 0.3877, 0.2827, 0.3886, 0.3585, 0.4036, 0.4036, 0.4018, 0.3769, 0.3998, 0.329, 0.3836, 0.3638, 0.3985, 0.3955, 0.3043, 0.4101, 0.4081, 0.4275, 0.4296, 0.3528, 0.4332, 0.3204, 0.4106, 0.4189, 0.441, 0.4225, 0.4462, 0.4298, 0.4443, 0.4405, 0.4044, 0.4038, 0.4523, 0.3392, 0.4112, 0.4161, 0.4394, 0.4528, 0.4568, 0.464, 0.4445, 0.4478, 0.4474, 0.4689, 0.4773, 0.4616, 0.4754]
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.001 1000 100 2 2 no fedavgfl 0.1 2 0.2 Skip 0.01 CIFAR100
#:test_acc_list [0.1178, 0.1453, 0.1608, 0.1727, 0.2219, 0.2261, 0.2067, 0.2347, 0.2297, 0.1795, 0.2558, 0.2603, 0.2682, 0.2917, 0.2942, 0.3064, 0.324, 0.2726, 0.2785, 0.2996, 0.3157, 0.3218, 0.3211, 0.3296, 0.356, 0.3465, 0.353, 0.3314, 0.3308, 0.3497, 0.3433, 0.3429, 0.3798, 0.3756, 0.3183, 0.3524, 0.3449, 0.3901, 0.3901, 0.3939, 0.3674, 0.3851, 0.4178, 0.4034, 0.3622, 0.3917, 0.4204, 0.3975, 0.4129, 0.4104, 0.4112, 0.4301, 0.3769, 0.4227, 0.4375, 0.4371, 0.431, 0.4134, 0.4325, 0.4234, 0.4361, 0.4518, 0.4502, 0.43, 0.423, 0.4398, 0.4483, 0.4014, 0.4301, 0.4262, 0.444, 0.4349, 0.4272, 0.4559, 0.4566, 0.4492, 0.4627, 0.4475, 0.4755, 0.4459, 0.478, 0.449, 0.4769, 0.4473, 0.4539, 0.482, 0.4617, 0.4872, 0.4667, 0.4656, 0.4841, 0.4811, 0.4792, 0.4853, 0.4912, 0.4837, 0.4895, 0.501, 0.4432, 0.4642]

#attack avg#
sh run_test_byz_p1.sh -1 CIFAR10 cnn 0.5 10 64 0.001 1000 100 2 2 trim_attack fedavgfl 0.1 2 0.2 Skip 0.01 CIFAR100
test_acc_list [0.1, 0.1, 0.1201, 0.1023, 0.1318, 0.1183, 0.2043, 0.213, 0.1259, 0.1778, 0.1471, 0.2007, 0.1521, 0.1524, 0.1775, 0.1913, 0.1917, 0.1906, 0.1387, 0.1006, 0.1043, 0.1466, 0.1777, 0.134, 0.1758, 0.1114, 0.2069, 0.1423, 0.1959, 0.1947, 0.1947, 0.1799, 0.2056, 0.2235, 0.2275, 0.1041, 0.1002, 0.1, 0.1012, 0.1317, 0.1516, 0.1525, 0.1442, 0.1045, 0.1276, 0.1637, 0.1569, 0.1729, 0.1462, 0.1337, 0.1171, 0.1042, 0.1142, 0.0999, 0.0998, 0.1, 0.1021, 0.0994, 0.0986, 0.1214, 0.1024, 0.1224, 0.1045, 0.1676, 0.1569, 0.1536, 0.1184, 0.169, 0.1817, 0.1113, 0.1164, 0.1756, 0.1875, 0.1903, 0.178, 0.1953, 0.1132, 0.1209, 0.1638, 0.1856, 0.1982, 0.2, 0.1217, 0.1274, 0.1336, 0.143, 0.1919, 0.135, 0.1499, 0.189, 0.1999, 0.1384, 0.1559, 0.196, 0.1509, 0.2029, 0.2061, 0.1968, 0.2187, 0.1879]
#fltrust file
###fltrust lr =0.001 seed =2 trim_attack
qql@qiangq:~/Documents/fltrust-download from Gong$ python3 test_byz_p.py
Iteration 09. Test_acc 0.1098
Iteration 19. Test_acc 0.1427
Iteration 29. Test_acc 0.1027
Iteration 39. Test_acc 0.1208
Iteration 49. Test_acc 0.1345
Iteration 59. Test_acc 0.1397
Iteration 69. Test_acc 0.1461
Iteration 79. Test_acc 0.1677
Iteration 89. Test_acc 0.1550
Iteration 99. Test_acc 0.1445
Iteration 109. Test_acc 0.1617
Iteration 119. Test_acc 0.1381
Iteration 129. Test_acc 0.1485
Iteration 139. Test_acc 0.1752
Iteration 149. Test_acc 0.1782
Iteration 159. Test_acc 0.2106
Iteration 169. Test_acc 0.1946
Iteration 179. Test_acc 0.1832
Iteration 189. Test_acc 0.1861
Iteration 199. Test_acc 0.2018
Iteration 209. Test_acc 0.1926
Iteration 219. Test_acc 0.2006
Iteration 229. Test_acc 0.2081
Iteration 239. Test_acc 0.1974
Iteration 249. Test_acc 0.2196
Iteration 259. Test_acc 0.2158
Iteration 269. Test_acc 0.2105
Iteration 279. Test_acc 0.2265
Iteration 289. Test_acc 0.2104
Iteration 299. Test_acc 0.2195
Iteration 309. Test_acc 0.2285
Iteration 319. Test_acc 0.2359
Iteration 329. Test_acc 0.2504
Iteration 339. Test_acc 0.2411
Iteration 349. Test_acc 0.2495
Iteration 359. Test_acc 0.2530
Iteration 369. Test_acc 0.2280
Iteration 379. Test_acc 0.2737
Iteration 389. Test_acc 0.2778
Iteration 399. Test_acc 0.2629
Iteration 409. Test_acc 0.2834
Iteration 419. Test_acc 0.2778
Iteration 429. Test_acc 0.2544
Iteration 439. Test_acc 0.2956
Iteration 449. Test_acc 0.2921
Iteration 459. Test_acc 0.3064
Iteration 469. Test_acc 0.2645
Iteration 479. Test_acc 0.3040
Iteration 489. Test_acc 0.2627
Iteration 499. Test_acc 0.3067
Iteration 509. Test_acc 0.3224
Iteration 519. Test_acc 0.3065
Iteration 529. Test_acc 0.3159
Iteration 539. Test_acc 0.3037
Iteration 549. Test_acc 0.3123
Iteration 559. Test_acc 0.3434
Iteration 569. Test_acc 0.3278
Iteration 579. Test_acc 0.3046
Iteration 589. Test_acc 0.3397
Iteration 599. Test_acc 0.3331
Iteration 609. Test_acc 0.3436
Iteration 619. Test_acc 0.3248
Iteration 629. Test_acc 0.3524
Iteration 639. Test_acc 0.3382
Iteration 649. Test_acc 0.3548
Iteration 659. Test_acc 0.3267
Iteration 669. Test_acc 0.3502
Iteration 679. Test_acc 0.3549
Iteration 689. Test_acc 0.3401
Iteration 699. Test_acc 0.3563
Iteration 709. Test_acc 0.3566
Iteration 719. Test_acc 0.3567
Iteration 729. Test_acc 0.3603
Iteration 739. Test_acc 0.3680
Iteration 749. Test_acc 0.3697
Iteration 759. Test_acc 0.3746
Iteration 769. Test_acc 0.3725
Iteration 779. Test_acc 0.3718
Iteration 789. Test_acc 0.3769
Iteration 799. Test_acc 0.3733
Iteration 809. Test_acc 0.3788
Iteration 819. Test_acc 0.3712
Iteration 829. Test_acc 0.3556
Iteration 839. Test_acc 0.3813
Iteration 849. Test_acc 0.3545
Iteration 859. Test_acc 0.3747
Iteration 869. Test_acc 0.3838
Iteration 879. Test_acc 0.3814
Iteration 889. Test_acc 0.3895
Iteration 899. Test_acc 0.3926
Iteration 909. Test_acc 0.4002
Iteration 919. Test_acc 0.3782
Iteration 929. Test_acc 0.4004
Iteration 939. Test_acc 0.3899
Iteration 949. Test_acc 0.3892
Iteration 959. Test_acc 0.4034
Iteration 969. Test_acc 0.4068
Iteration 979. Test_acc 0.3998
Iteration 989. Test_acc 0.4079
Iteration 999. Test_acc 0.3951

###fltrust lr=0.005 seed=2 noattack
qql@qiangq:~/Documents/fltrust-download from Gong$ python3 test_byz_p.py
Iteration 09. Test_acc 0.1020
Iteration 19. Test_acc 0.1273
Iteration 29. Test_acc 0.1621
Iteration 39. Test_acc 0.1405
Iteration 49. Test_acc 0.1779
Iteration 59. Test_acc 0.1644
Iteration 69. Test_acc 0.1967
Iteration 79. Test_acc 0.1936
Iteration 89. Test_acc 0.1040
Iteration 99. Test_acc 0.2418
Iteration 109. Test_acc 0.1921
Iteration 119. Test_acc 0.2716
Iteration 129. Test_acc 0.1458
Iteration 139. Test_acc 0.1773
Iteration 149. Test_acc 0.1833
Iteration 159. Test_acc 0.2014
Iteration 169. Test_acc 0.2433
Iteration 179. Test_acc 0.2182
Iteration 189. Test_acc 0.2776
Iteration 199. Test_acc 0.2710
Iteration 209. Test_acc 0.2522
Iteration 219. Test_acc 0.2117
Iteration 229. Test_acc 0.1480
Iteration 239. Test_acc 0.2322
Iteration 249. Test_acc 0.2777
Iteration 259. Test_acc 0.1809
Iteration 269. Test_acc 0.2942
Iteration 279. Test_acc 0.2364
Iteration 289. Test_acc 0.2463
Iteration 299. Test_acc 0.2002
Iteration 309. Test_acc 0.2323
Iteration 319. Test_acc 0.1765
Iteration 329. Test_acc 0.2540
Iteration 339. Test_acc 0.1963
Iteration 349. Test_acc 0.1904
Iteration 359. Test_acc 0.2544
Iteration 369. Test_acc 0.2681
Iteration 379. Test_acc 0.2641
Iteration 389. Test_acc 0.2318
Iteration 399. Test_acc 0.2760
Iteration 409. Test_acc 0.2518
Iteration 419. Test_acc 0.2628
Iteration 429. Test_acc 0.2796
Iteration 439. Test_acc 0.2957
Iteration 449. Test_acc 0.3078
Iteration 459. Test_acc 0.3283
Iteration 469. Test_acc 0.3416
Iteration 479. Test_acc 0.3434
Iteration 489. Test_acc 0.2457
Iteration 499. Test_acc 0.3090
Iteration 509. Test_acc 0.3433
Iteration 519. Test_acc 0.2619
Iteration 529. Test_acc 0.3483
Iteration 539. Test_acc 0.3404
Iteration 549. Test_acc 0.2564
Iteration 559. Test_acc 0.3012
Iteration 569. Test_acc 0.3174
Iteration 579. Test_acc 0.3210
Iteration 589. Test_acc 0.3392
Iteration 599. Test_acc 0.2936
Iteration 609. Test_acc 0.3115
Iteration 619. Test_acc 0.3478
Iteration 629. Test_acc 0.3725
Iteration 639. Test_acc 0.3639
Iteration 649. Test_acc 0.3867
Iteration 659. Test_acc 0.3757
Iteration 669. Test_acc 0.3935
Iteration 679. Test_acc 0.4119
Iteration 689. Test_acc 0.3412
Iteration 699. Test_acc 0.3800
Iteration 709. Test_acc 0.3893
Iteration 719. Test_acc 0.3761
Iteration 729. Test_acc 0.4181
Iteration 739. Test_acc 0.3757
Iteration 749. Test_acc 0.4300
Iteration 759. Test_acc 0.4244
Iteration 769. Test_acc 0.3732
Iteration 779. Test_acc 0.4084
Iteration 789. Test_acc 0.4206
Iteration 799. Test_acc 0.4125
Iteration 809. Test_acc 0.3445
Iteration 819. Test_acc 0.4282
Iteration 829. Test_acc 0.4293
Iteration 839. Test_acc 0.3880
Iteration 849. Test_acc 0.4350
Iteration 859. Test_acc 0.4362
Iteration 869. Test_acc 0.4273
Iteration 879. Test_acc 0.4509
Iteration 889. Test_acc 0.4452
Iteration 899. Test_acc 0.4326
Iteration 909. Test_acc 0.4039
Iteration 919. Test_acc 0.4714
Iteration 929. Test_acc 0.4017
Iteration 939. Test_acc 0.4327
Iteration 949. Test_acc 0.4408
Iteration 959. Test_acc 0.4613
Iteration 969. Test_acc 0.4561
Iteration 979. Test_acc 0.4351
Iteration 989. Test_acc 0.4753
Iteration 999. Test_acc 0.4767

