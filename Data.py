import matplotlib.pyplot as plt

from functions import *


def plot_weak_errors_of_various_quadrature_rules():
    modes = ['GG', 'NGG', 'OL1', 'OLD', 'OL2', 'BL2', 'AE', 'AK']
    H = np.array([-0.1, 0.001, 0.01, 0.1])
    N = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    errors_smile = np.empty((len(modes), len(H), len(N)))
    errors_smile[0, 0, :] = np.array(
        [0.2993, 0.1835, 0.1335, 0.1058, 0.08802, 0.06109, 0.03707, 0.03697, 0.03844, 0.02482])
    errors_smile[0, 1, :] = np.array([0.1829, 0.1155, 0.08704, 0.07066, 0.07599, 0.03161, 0.01965, 0.01898, 0.01932,
                                      0.01263])
    errors_smile[0, 2, :] = np.array([0.1746, 0.1106, 0.08365, 0.06806, 0.07168, 0.02973, 0.01853, 0.01785, 0.01814,
                                      0.01187])
    errors_smile[0, 3, :] = np.array([0.1343, 0.08288, 0.06017, 0.04405, 0.05058, 0.02121, 0.01371, 0.01245, 0.01206,
                                      0.008041])
    errors_smile[1, 0, :] = np.array([0.3186, 0.2182, 0.2346, 0.2014, 0.1584, 0.1338, 0.1178, 0.1178, 0.1178, 0.07052])
    errors_smile[1, 1, :] = np.array(
        [0.1938, 0.1416, 0.1575, 0.1316, 0.1074, 0.1083, 0.07282, 0.07282, 0.07282, 0.04476])
    errors_smile[1, 2, :] = np.array(
        [0.1850, 0.1359, 0.1514, 0.1263, 0.1033, 0.1043, 0.06962, 0.06962, 0.06962, 0.04283])
    errors_smile[1, 3, :] = np.array([0.1477, 0.1067, 0.1231, 0.09812, 0.06501, 0.09107, 0.05525, 0.05525, 0.05525,
                                      0.03414])
    errors_smile[2, 0, :] = np.array(
        [0.1493, 0.05526, 0.02527, 0.01281, 0.006811, 0.003993, 0.002384, 0.001475, 0.0007887,
         0.0004666])
    errors_smile[2, 1, :] = np.array([0.08538, 0.02818, 0.01147, 0.005254, 0.002617, 0.001383, 0.0007650, 0.0004390,
                                      0.0002604, 0.0001227])
    errors_smile[2, 2, :] = np.array([0.08398, 0.02640, 0.01066, 0.004837, 0.002280, 0.001252, 0.0005839, 0.0003911,
                                      0.0001781, 0.00009751])
    errors_smile[2, 3, :] = np.array([0.07348, 0.02050, 0.007378, 0.003057, 0.001399, 0.0006876, 0.0002882, 0.0001918,
                                      0.00008087, 0.00003976])
    errors_smile[3, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_smile[3, 1, :] = np.array([0.1051, 0.1051, 0.1051, 0.1051, 0.1051, 0.1051, 0.1051, 0.1051, 0.1051, 0.1051])
    errors_smile[3, 2, :] = np.array(
        [0.1059, 0.1059, 0.1059, 0.1057, 0.1048, 0.1030, 0.1001, 0.09630, 0.09177, 0.08676])
    errors_smile[3, 3, :] = np.array([0.09957, 0.09904, 0.08498, 0.05636, 0.03110, 0.01620, 0.01132, 0.01207, 0.01391,
                                      0.01479])
    errors_smile[4, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_smile[4, 1, :] = np.array([0.08315, 0.07133, 0.06394, 0.05848, 0.05395, 0.05051, 0.04740, 0.04469, 0.04228,
                                      0.04010])
    errors_smile[4, 2, :] = np.array(
        [0.04000, 0.02559, 0.01816, 0.01362, 0.01039, 0.008078, 0.007067, 0.007123, 0.007065,
         0.006933])
    errors_smile[4, 3, :] = np.array([0.008936, 0.006503, 0.005534, 0.004271, 0.003193, 0.002359, 0.001735, 0.001276,
                                      0.0009414, 0.0006972])
    errors_smile[5, 0, :] = np.array([0.1493, 0.005914, 0.001116, 0.0001230, 0.00002444, 0.000003325, 0.000002383,
                                      0.000009124, 0.00002084, 0.00003136])
    errors_smile[5, 1, :] = np.array([0.08315, 0.002227, 0.001009, 0.00007388, 0.00001411, 0.000002441, 0.000002527,
                                      0.000002203, 0.000002100, 0.000002111])
    errors_smile[5, 2, :] = np.array([0.04000, 0.004305, 0.0009287, 0.00007170, 0.00001559, 0.000006044, 0.000004745,
                                      0.000004040, 0.000003861, 0.000003813])
    errors_smile[5, 3, :] = np.array([0.008936, 0.004424, 0.0006565, 0.00004705, 0.000006326, 0.000001694, 0.0000009140,
                                      0.000001674, 0.000001204, 0.0000009833])
    errors_smile[6, 0, :] = np.array([0.4944, 0.3977, 0.3446, 0.3101, 0.2856, 0.2672, 0.2527, 0.2407, 0.2307, 0.2222])
    errors_smile[6, 1, :] = np.array([0.3140, 0.2507, 0.2156, 0.1928, 0.1765, 0.1641, 0.1543, 0.1463, 0.1395, 0.1338])
    errors_smile[6, 2, :] = np.array([0.3009, 0.2404, 0.2102, 0.1894, 0.1741, 0.1621, 0.1523, 0.1442, 0.1373, 0.1314])
    errors_smile[6, 3, :] = np.array([0.2436, 0.2007, 0.1760, 0.1592, 0.1467, 0.1368, 0.1288, 0.1221, 0.1164, 0.1115])
    errors_smile[7, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_smile[7, 1, :] = np.array([np.nan, 0.08138, 0.08138, 0.2438, 0.2438, 0.2950, 0.2950, 0.3127, 0.3127, 0.3054])
    errors_smile[7, 2, :] = np.array(
        [np.nan, 0.05366, 0.05366, 0.09091, 0.09091, 0.07631, 0.07631, 0.06857, 0.06857, 0.05470])
    errors_smile[7, 3, :] = np.array([np.nan, 0.01112, 0.01112, 0.02004, 0.02004, 0.01659, 0.01659, 0.009991, 0.009991,
                                      0.005032])

    errors_surface = np.empty((len(modes), len(H), len(N)))
    errors_surface[0, 0, :] = np.array(
        [0.3676, 0.2184, 0.1814, 0.1468, 0.1238, 0.1112, 0.08756, 0.07604, 0.06933, 0.05121])
    errors_surface[0, 1, :] = np.array([0.2671, 0.1734, 0.1512, 0.1226, 0.1243, 0.08618, 0.07406, 0.05997, 0.05077,
                                        0.03829])
    errors_surface[0, 2, :] = np.array([0.2627, 0.1727, 0.1513, 0.1231, 0.1231, 0.08671, 0.07487, 0.06047, 0.05102,
                                        0.03852])
    errors_surface[0, 3, :] = np.array([0.2142, 0.1558, 0.1422, 0.1227, 0.1167, 0.08732, 0.07845, 0.06253, 0.05148,
                                        0.03940])
    errors_surface[1, 0, :] = np.array(
        [0.4053, 0.2304, 0.1814, 0.1381, 0.1115, 0.09455, 0.1084, 0.1084, 0.1084, 0.05967])
    errors_surface[1, 1, :] = np.array([0.2943, 0.1635, 0.1287, 0.09995, 0.08213, 0.1190, 0.06733, 0.06733, 0.06733,
                                        0.03557])
    errors_surface[1, 2, :] = np.array([0.2895, 0.1584, 0.1247, 0.09702, 0.07992, 0.1172, 0.06624, 0.06624, 0.06624,
                                        0.03413])
    errors_surface[1, 3, :] = np.array([0.2368, 0.1363, 0.1109, 0.08611, 0.1972, 0.09581, 0.05410, 0.05410, 0.05410,
                                        0.02831])
    errors_surface[2, 0, :] = np.array(
        [0.2321, 0.09025, 0.04975, 0.02881, 0.01620, 0.01043, 0.007202, 0.004336, 0.002289,
         0.001360])
    errors_surface[2, 1, :] = np.array(
        [0.1954, 0.08027, 0.03203, 0.01583, 0.009003, 0.005313, 0.003383, 0.001956, 0.001167,
         0.0005538])
    errors_surface[2, 2, :] = np.array([0.1933, 0.08020, 0.03161, 0.01552, 0.008376, 0.005155, 0.002769, 0.001885,
                                        0.0008742, 0.0004845])
    errors_surface[2, 3, :] = np.array([0.1653, 0.07787, 0.02632, 0.01233, 0.006758, 0.003720, 0.001884, 0.001264,
                                        0.0005309, 0.0002581])
    errors_surface[3, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_surface[3, 1, :] = np.array([0.2726, 0.3082, 0.3616, 0.3856, 0.4079, 0.4245, 0.4475, 0.4475, 0.4475, 0.4474])
    errors_surface[3, 2, :] = np.array([0.2698, 0.3049, 0.3573, 0.3789, 0.3906, 0.3820, 0.3574, 0.3230, 0.2834, 0.2422])
    errors_surface[3, 3, :] = np.array([0.2331, 0.2597, 0.2062, 0.09044, 0.06149, 0.07431, 0.06183, 0.04812, 0.04128,
                                        0.03942])
    errors_surface[4, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_surface[4, 1, :] = np.array([0.2378, 0.1904, 0.1452, 0.1188, 0.1455, 0.1252, 0.1600, 0.1602, 0.1600, 0.1596])
    errors_surface[4, 2, :] = np.array([0.1977, 0.1613, 0.1096, 0.08941, 0.09913, 0.09864, 0.1056, 0.09607, 0.08691,
                                        0.07817])
    errors_surface[4, 3, :] = np.array(
        [0.1775, 0.1330, 0.06808, 0.03679, 0.01505, 0.006230, 0.004136, 0.002843, 0.002889,
         0.002282])
    errors_surface[5, 0, :] = np.array([0.2321, 0.07255, 0.008061, 0.006436, 0.001758, 0.0001180, 0.0001224, 0.00001796,
                                        0.00001086, 0.000008811])
    errors_surface[5, 1, :] = np.array([0.2378, 0.04223, 0.009478, 0.01044, 0.002738, 0.0005558, 0.0001236, 0.00001813,
                                        0.00001460, 0.000009947])
    errors_surface[5, 2, :] = np.array([0.1977, 0.02710, 0.009116, 0.008241, 0.002668, 0.0001901, 0.0001916, 0.00006338,
                                        0.00001990, 0.000006955])
    errors_surface[5, 3, :] = np.array([0.1775, 0.06111, 0.01012, 0.008072, 0.001943, 0.0003600, 0.00008426, 0.00002153,
                                        0.000003113, 0.000002971])
    errors_surface[6, 0, :] = np.array([0.5290, 0.5121, 0.5276, 0.5302, 0.5361, 0.5404, 0.5516, 0.5458, 0.5402, 0.5350])
    errors_surface[6, 1, :] = np.array([0.4062, 0.3877, 0.4045, 0.4074, 0.4142, 0.4194, 0.4336, 0.4260, 0.4192, 0.4129])
    errors_surface[6, 2, :] = np.array([0.4006, 0.3823, 0.3990, 0.4019, 0.4086, 0.4137, 0.4278, 0.4203, 0.4135, 0.4073])
    errors_surface[6, 3, :] = np.array([0.3385, 0.3223, 0.3371, 0.3307, 0.3457, 0.3503, 0.3631, 0.3562, 0.3501, 0.3445])
    errors_surface[7, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_surface[7, 1, :] = np.array([np.nan, 0.1702, 0.1910, 0.2445, 0.2641, 0.2659, 0.2698, 0.2656, 0.2656, 0.2656])
    errors_surface[7, 2, :] = np.array([np.nan, 0.1369, 0.1469, 0.1537, 0.1566, 0.1549, 0.1598, 0.1418, 0.1418, 0.1290])
    errors_surface[7, 3, :] = np.array([np.nan, 0.03945, 0.04089, 0.02426, 0.02153, 0.01218, 0.01473, 0.01197, 0.01197,
                                        0.009941])

    errors_skew = np.empty((len(modes), len(H), len(N)))
    errors_skew[0, 0, :] = np.array([0.5949, 0.4546, 0.3109, 0.2606, 0.2312, 0.2067, 0.1989, 0.1593, 0.1312, 0.1001])
    errors_skew[0, 1, :] = np.array([0.6281, 0.4965, 0.4558, 0.4080, 0.4057, 0.3406, 0.3468, 0.2653, 0.2097, 0.1637])
    errors_skew[0, 2, :] = np.array([0.6314, 0.5007, 0.4666, 0.4206, 0.4189, 0.3549, 0.3625, 0.2785, 0.2195, 0.1710])
    errors_skew[0, 3, :] = np.array([0.6658, 0.5413, 0.5239, 0.5022, 0.5015, 0.4581, 0.4736, 0.3969, 0.3266, 0.2602])
    errors_skew[1, 0, :] = np.array([0.6563, 0.4560, 0.2854, 0.1942, 0.1551, 0.1226, 0.2372, 0.2372, 0.2372, 0.1126])
    errors_skew[1, 1, :] = np.array([0.6568, 0.4865, 0.3610, 0.2514, 0.2039, 0.4286, 0.2760, 0.2760, 0.2760, 0.1274])
    errors_skew[1, 2, :] = np.array([0.6557, 0.4928, 0.3669, 0.2568, 0.2086, 0.4322, 0.2794, 0.2794, 0.2794, 0.1286])
    errors_skew[1, 3, :] = np.array([0.6238, 0.5259, 0.4041, 0.2989, 0.6589, 0.4424, 0.3032, 0.3032, 0.3032, 0.1370])
    errors_skew[2, 0, :] = np.array([0.4971, 0.3113, 0.1343, 0.06319, 0.03171, 0.02263, 0.01952, 0.01464, 0.01135,
                                     0.009989])
    errors_skew[2, 1, :] = np.array([0.5386, 0.3443, 0.1395, 0.06123, 0.03103, 0.02197, 0.01494, 0.009043, 0.004978,
                                     0.002309])
    errors_skew[2, 2, :] = np.array([0.5425, 0.3472, 0.1398, 0.06100, 0.02960, 0.02206, 0.01229, 0.009011, 0.003919,
                                     0.002099])
    errors_skew[2, 3, :] = np.array([0.5784, 0.3709, 0.1409, 0.06287, 0.03566, 0.02153, 0.01355, 0.008213, 0.003442,
                                     0.001881])
    errors_skew[3, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_skew[3, 1, :] = np.array([0.6330, 0.7267, 0.8299, 0.8652, 0.8931, 0.9111, 0.9326, 0.9325, 0.9324, 0.9322])
    errors_skew[3, 2, :] = np.array([0.6354, 0.7268, 0.8280, 0.8561, 0.8476, 0.7747, 0.6825, 0.5963, 0.5176, 0.4488])
    errors_skew[3, 3, :] = np.array([0.6298, 0.6974, 0.4783, 0.2126, 0.2893, 0.2087, 0.1534, 0.1612, 0.1336, 0.1091])
    errors_skew[4, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_skew[4, 1, :] = np.array([0.6964, 0.6005, 0.4239, 0.3763, 0.5524, 0.4648, 0.5973, 0.5808, 0.5643, 0.5478])
    errors_skew[4, 2, :] = np.array([0.6507, 0.5371, 0.3431, 0.3215, 0.2737, 0.2319, 0.1979, 0.1736, 0.1533, 0.1365])
    errors_skew[4, 3, :] = np.array([0.6465, 0.5013, 0.2680, 0.1551, 0.05940, 0.02618, 0.01424, 0.009396, 0.006557,
                                     0.006901])
    errors_skew[5, 0, :] = np.array([0.4971, 0.1577, 0.03803, 0.01572, 0.01015, 0.008630, 0.009321, 0.009907, 0.009898,
                                     0.007654])
    errors_skew[5, 1, :] = np.array(
        [0.6964, 0.2493, 0.04381, 0.06475, 0.02578, 0.003323, 0.002775, 0.0009356, 0.0004261,
         0.0001619])
    errors_skew[5, 2, :] = np.array(
        [0.6507, 0.09671, 0.04617, 0.06568, 0.02506, 0.003042, 0.002976, 0.0009268, 0.0004388,
         0.0001734])
    errors_skew[5, 3, :] = np.array(
        [0.6465, 0.1656, 0.04984, 0.04963, 0.01873, 0.007915, 0.002388, 0.0009434, 0.0003213,
         0.0001135])
    errors_skew[6, 0, :] = np.array([0.8370, 0.8368, 0.8859, 0.9006, 0.9161, 0.9265, 0.9432, 0.9394, 0.9359, 0.9326])
    errors_skew[6, 1, :] = np.array([0.8189, 0.8186, 0.8652, 0.8798, 0.8955, 0.9065, 0.9245, 0.9204, 0.9165, 0.9130])
    errors_skew[6, 2, :] = np.array([0.8165, 0.8163, 0.8627, 0.8773, 0.8931, 0.9041, 0.9223, 0.9181, 0.9142, 0.9106])
    errors_skew[6, 3, :] = np.array([0.7791, 0.7788, 0.8255, 0.8408, 0.8577, 0.8698, 0.8903, 0.8855, 0.8811, 0.8770])
    errors_skew[7, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_skew[7, 1, :] = np.array([np.nan, 0.4161, 0.3536, 0.4026, 0.4281, 0.4637, 0.4742, 0.4906, 0.4906, 0.5206])
    errors_skew[7, 2, :] = np.array([np.nan, 0.2921, 0.2651, 0.2797, 0.2845, 0.2749, 0.2897, 0.2560, 0.2560, 0.2312])
    errors_skew[7, 3, :] = np.array(
        [np.nan, 0.2504, 0.3797, 0.05494, 0.05192, 0.02523, 0.04037, 0.03172, 0.03172, 0.02682])

    errors_dig = np.empty((len(modes), len(H), len(N)))
    errors_dig[0, 0, :] = np.array([0.8047, 0.6739, 0.5668, 0.4932, 0.4394, 0.3033, 0.1885, 0.1948, 0.2076, 0.1351])
    errors_dig[0, 1, :] = np.array([0.6838, 0.5479, 0.4665, 0.4146, 0.3735, 0.1701, 0.1024, 0.1054, 0.1131, 0.07230])
    errors_dig[0, 2, :] = np.array([0.6726, 0.5381, 0.4593, 0.4093, 0.3578, 0.1610, 0.09681, 0.09967, 0.1070, 0.06840])
    errors_dig[0, 3, :] = np.array(
        [0.5631, 0.4522, 0.3985, 0.1811, 0.2215, 0.09010, 0.05415, 0.05606, 0.06072, 0.03940])
    errors_dig[1, 0, :] = np.array([0.8095, 0.6614, 0.5492, 0.4743, 0.4271, 0.3970, 0.1922, 0.1922, 0.1922, 0.1275])
    errors_dig[1, 1, :] = np.array([0.6786, 0.5235, 0.4382, 0.3905, 0.3633, 0.1660, 0.1008, 0.1008, 0.1008, 0.06895])
    errors_dig[1, 2, :] = np.array([0.6660, 0.5123, 0.4300, 0.3845, 0.3587, 0.1568, 0.09524, 0.09524, 0.09524, 0.06555])
    errors_dig[1, 3, :] = np.array(
        [0.5389, 0.4128, 0.3593, 0.3326, 0.1609, 0.08656, 0.05488, 0.05488, 0.05488, 0.04075])
    errors_dig[2, 0, :] = np.array([0.4531, 0.2305, 0.1167, 0.06180, 0.02255, 0.01990, 0.01196, 0.007390, 0.003970,
                                    0.002358])
    errors_dig[2, 1, :] = np.array([0.2559, 0.1117, 0.04918, 0.02319, 0.01171, 0.006236, 0.003467, 0.001997, 0.001189,
                                    0.0005621])
    errors_dig[2, 2, :] = np.array([0.2408, 0.1039, 0.04527, 0.02114, 0.01009, 0.005592, 0.002622, 0.001770, 0.0008155,
                                    0.0004527])
    errors_dig[2, 3, :] = np.array(
        [0.1148, 0.04627, 0.01815, 0.007749, 0.003599, 0.001783, 0.0007461, 0.0005012, 0.0002103,
         0.0001035])
    errors_dig[3, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_dig[3, 1, :] = np.array([0.6715, 0.6715, 0.6715, 0.6715, 0.6715, 0.6715, 0.6715, 0.6715, 0.6715, 0.6714])
    errors_dig[3, 2, :] = np.array([0.6638, 0.6638, 0.6636, 0.6607, 0.6496, 0.6262, 0.5901, 0.5438, 0.4981, 0.4348])
    errors_dig[3, 3, :] = np.array(
        [0.5668, 0.5607, 0.4094, 0.1736, 0.05306, 0.02455, 0.03151, 0.05247, 0.05776, 0.05011])
    errors_dig[4, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_dig[4, 1, :] = np.array([0.6289, 0.5683, 0.5266, 0.4938, 0.9642, 0.9580, 0.4218, 0.4030, 0.3859, 0.3702])
    errors_dig[4, 2, :] = np.array(
        [0.3713, 0.2589, 0.1884, 0.1383, 0.1004, 0.07093, 0.04741, 0.02840, 0.01576, 0.008926])
    errors_dig[4, 3, :] = np.array([0.02453, 0.06922, 0.06526, 0.05242, 0.03987, 0.02965, 0.02185, 0.01607, 0.01184,
                                    0.008771])
    errors_dig[5, 0, :] = np.array([0.4531, 0.06368, 0.01277, 0.001581, 0.0002547, 0.00003122, 0.00002112, 0.00002308,
                                    0.00001572, 0.000009836])
    errors_dig[5, 1, :] = np.array(
        [0.6289, 0.02437, 0.01109, 0.0008605, 0.00008251, 0.00001374, 0.000007155, 0.000002918,
         0.000002170, 0.000001953])
    errors_dig[5, 2, :] = np.array(
        [0.3713, 0.04769, 0.01024, 0.0009368, 0.00008237, 0.00003030, 0.000005892, 0.000002482,
         0.000002033, 0.000001876])
    errors_dig[5, 3, :] = np.array(
        [0.02453, 0.03563, 0.008303, 0.0006318, 0.00005841, 0.00001105, 0.000003487, 0.000002193,
         0.000002196, 0.000002195])
    errors_dig[6, 0, :] = np.array([0.9074, 0.8596, 0.8159, 0.7794, 0.7491, 0.7236, 0.7020, 0.6833, 0.6670, 0.6527])
    errors_dig[6, 1, :] = np.array([0.8513, 0.7598, 0.6900, 0.6367, 0.5953, 0.5622, 0.5353, 0.5129, 0.4939, 0.4777])
    errors_dig[6, 2, :] = np.array([0.8446, 0.7493, 0.6775, 0.6232, 0.5811, 0.5477, 0.5205, 0.4980, 0.4790, 0.4628])
    errors_dig[6, 3, :] = np.array([0.7610, 0.6337, 0.5480, 0.4872, 0.4423, 0.4080, 0.3809, 0.3590, 0.3410, 0.3260])
    errors_dig[7, 0, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    errors_dig[7, 1, :] = np.array([np.nan, 0.2205, 0.2205, 0.5885, 0.5885, 0.9022, 0.9022, 1.106, 1.106, 1.269])
    errors_dig[7, 2, :] = np.array([np.nan, 0.1812, 0.1812, 0.5110, 0.5110, 0.7270, 0.7270, 0.8455, 0.8455, 0.9225])
    errors_dig[7, 3, :] = np.array([np.nan, 0.05992, 0.05992, 0.1222, 0.1222, 0.1387, 0.1387, 0.1397, 0.1397, 0.1347])

    for i in range(errors_smile.shape[0]):
        if i != 3:
            plt.loglog(N, errors_smile[i, 1, :], color=color(i, errors_smile.shape[0]), label=modes[i])
    plt.loglog(N, 2e-05 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of implied volatility smiles for $H=0.001$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.loglog(N, errors_smile[0, 0, :], color=color(0, 4), label='GG')
    plt.loglog(N, errors_smile[0, 1, :], '--', color=color(0, 4))
    plt.loglog(N, errors_smile[0, 3, :], '-.', color=color(0, 4))
    plt.loglog(N, errors_smile[2, 0, :], color=color(1, 4), label='OL1')
    plt.loglog(N, errors_smile[2, 1, :], '--', color=color(1, 4))
    plt.loglog(N, errors_smile[2, 3, :], '-.', color=color(1, 4))
    plt.loglog(N, errors_smile[4, 0, :], color=color(2, 4), label='OL2')
    plt.loglog(N, errors_smile[4, 1, :], '--', color=color(2, 4))
    plt.loglog(N, errors_smile[4, 3, :], '-.', color=color(2, 4))
    plt.loglog(N, errors_smile[5, 0, :], color=color(3, 4), label='BL2')
    plt.loglog(N, errors_smile[5, 1, :], '--', color=color(3, 4))
    plt.loglog(N, errors_smile[5, 3, :], '-.', color=color(3, 4))
    plt.loglog(N, 2e-05 * np.ones(len(N)), 'k--')
    plt.loglog(N, 2e-04 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of implied volatility smiles')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    for i in range(errors_surface.shape[0]):
        if i != 3:
            plt.loglog(N, errors_surface[i, 1, :], color=color(i, errors_surface.shape[0]), label=modes[i])
    plt.loglog(N, 2e-05 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of implied volatility surfaces for $H=0.001$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.loglog(N, errors_surface[0, 0, :], color=color(0, 4), label='GG')
    plt.loglog(N, errors_surface[0, 1, :], '--', color=color(0, 4))
    plt.loglog(N, errors_surface[0, 3, :], '-.', color=color(0, 4))
    plt.loglog(N, errors_surface[2, 0, :], color=color(1, 4), label='OL1')
    plt.loglog(N, errors_surface[2, 1, :], '--', color=color(1, 4))
    plt.loglog(N, errors_surface[2, 3, :], '-.', color=color(1, 4))
    plt.loglog(N, errors_surface[4, 0, :], color=color(2, 4), label='OL2')
    plt.loglog(N, errors_surface[4, 1, :], '--', color=color(2, 4))
    plt.loglog(N, errors_surface[4, 3, :], '-.', color=color(2, 4))
    plt.loglog(N, errors_surface[5, 0, :], color=color(3, 4), label='BL2')
    plt.loglog(N, errors_surface[5, 1, :], '--', color=color(3, 4))
    plt.loglog(N, errors_surface[5, 3, :], '-.', color=color(3, 4))
    plt.loglog(N, 2e-05 * np.ones(len(N)), 'k--')
    plt.loglog(N, 2e-04 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of implied volatility surfaces')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    for i in range(errors_skew.shape[0]):
        if i != 3:
            plt.loglog(N, errors_skew[i, 1, :], color=color(i, errors_skew.shape[0]), label=modes[i])
    plt.loglog(N, 2e-04 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of the skew for $H=0.001$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.loglog(N, errors_skew[0, 0, :], color=color(0, 4), label='GG')
    plt.loglog(N, errors_skew[0, 1, :], '--', color=color(0, 4))
    plt.loglog(N, errors_skew[0, 3, :], '-.', color=color(0, 4))
    plt.loglog(N, errors_skew[2, 0, :], color=color(1, 4), label='OL1')
    plt.loglog(N, errors_skew[2, 1, :], '--', color=color(1, 4))
    plt.loglog(N, errors_skew[2, 3, :], '-.', color=color(1, 4))
    plt.loglog(N, errors_skew[4, 0, :], color=color(2, 4), label='OL2')
    plt.loglog(N, errors_skew[4, 1, :], '--', color=color(2, 4))
    plt.loglog(N, errors_skew[4, 3, :], '-.', color=color(2, 4))
    plt.loglog(N, errors_skew[5, 0, :], color=color(3, 4), label='BL2')
    plt.loglog(N, errors_skew[5, 1, :], '--', color=color(3, 4))
    plt.loglog(N, errors_skew[5, 3, :], '-.', color=color(3, 4))
    plt.loglog(N, 2e-04 * np.ones(len(N)), 'k--')
    plt.loglog(N, 0.01 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of the skew')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    for i in range(errors_dig.shape[0]):
        if i != 3:
            plt.loglog(N, errors_dig[i, 1, :], color=color(i, errors_dig.shape[0]), label=modes[i])
    plt.loglog(N, 2e-05 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of digital European call option for $H=0.001$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.loglog(N, errors_dig[0, 0, :], color=color(0, 4), label='GG')
    plt.loglog(N, errors_dig[0, 1, :], '--', color=color(0, 4))
    plt.loglog(N, errors_dig[0, 3, :], '-.', color=color(0, 4))
    plt.loglog(N, errors_dig[2, 0, :], color=color(1, 4), label='OL1')
    plt.loglog(N, errors_dig[2, 1, :], '--', color=color(1, 4))
    plt.loglog(N, errors_dig[2, 3, :], '-.', color=color(1, 4))
    plt.loglog(N, errors_dig[4, 0, :], color=color(2, 4), label='OL2')
    plt.loglog(N, errors_dig[4, 1, :], '--', color=color(2, 4))
    plt.loglog(N, errors_dig[4, 3, :], '-.', color=color(2, 4))
    plt.loglog(N, errors_dig[5, 0, :], color=color(3, 4), label='BL2')
    plt.loglog(N, errors_dig[5, 1, :], '--', color=color(3, 4))
    plt.loglog(N, errors_dig[5, 3, :], '-.', color=color(3, 4))
    plt.loglog(N, 2e-05 * np.ones(len(N)), 'k-')
    plt.xlabel('Number of nodes N')
    plt.title(r'Relative errors of digital European call option')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_computed_skews():
    # Skews for the Markovian approximations of rough Heston paper. The parameters used are
    # lambda_ = 0.3, nu = 0.3, theta = 0.02, V_0 = 0.02, rho = -0.7, H = -0.1, rel_tol = 5e-03,
    # T = log_linspace(0.004, 1, 25)
    T = log_linspace(0.004, 1, 25)
    skew_true = np.array([7.76227043, 6.83460674, 6.06014079, 5.36707535, 4.72195099,
                          4.14050653, 3.67066926, 3.21141756, 2.82544305, 2.49308724,
                          2.18138936, 1.92299924, 1.677964, 1.47737222, 1.28519484,
                          1.12974831, 0.97946874, 0.85928106, 0.74218297, 0.64461797,
                          0.55868217, 0.48349051, 0.41754039, 0.35980137, 0.30936484])
    skew_GG_1 = np.array([3.17194924, 3.1372388, 3.07091247, 2.96618519, 2.82267077,
                          2.64466045, 2.4410288, 2.22218506, 1.99848156, 1.77781554,
                          1.56632214, 1.36803063, 1.18515886, 1.01882245, 0.86921192,
                          0.73631823, 0.61909992, 0.51746611, 0.42921165, 0.35385757,
                          0.29003174, 0.23650808, 0.19205322, 0.15537428, 0.12531105])
    skew_BL_1 = np.array([4.87932643, 4.67447102, 4.41087973, 4.10456035, 3.77111889,
                          3.42637565, 3.08315363, 2.75218111, 2.43902185, 2.14739212,
                          1.87944775, 1.63506515, 1.41424012, 1.21573878, 1.03876675,
                          0.88180743, 0.74369235, 0.6237971, 0.51968804, 0.43025239,
                          0.35440183, 0.2903038, 0.23672276, 0.19223989, 0.1555746])
    skew_GG_2 = np.array([5.28730569, 4.85391903, 4.41667997, 3.99276296, 3.59180405,
                          3.21953722, 2.87926971, 2.56943117, 2.28759045, 2.03127658,
                          1.79674138, 1.58155101, 1.38415772, 1.2036754, 1.03966044,
                          0.89180772, 0.76006435, 0.64318158, 0.54076503, 0.45161382,
                          0.37496299, 0.30941687, 0.25396824, 0.20745775, 0.16873979])
    skew_BL_2 = np.array([7.95307153, 7.03451809, 6.20785592, 5.46818735, 4.80618691,
                          4.21760225, 3.69460116, 3.23354761, 2.82795714, 2.47430206,
                          2.16698947, 1.90158033, 1.67300249, 1.47668471, 1.30746701,
                          1.16050766, 1.03133172, 0.91650667, 0.81269883, 0.7181003,
                          0.63157814, 0.55258828, 0.48087949, 0.41616754, 0.35814053])
    skew_GG_3 = np.array([5.32684367, 4.90291067, 4.47257561, 4.05334853, 3.65485717,
                          3.2833474, 2.94123756, 2.62882723, 2.34424333, 2.08556893,
                          1.85099612, 1.63847625, 1.44667611, 1.27349751, 1.11738211,
                          0.97622302, 0.84861192, 0.73359147, 0.62996581, 0.53749124,
                          0.4554101, 0.3833938, 0.32063007, 0.26640496, 0.22009299])
    skew_BL_3 = np.array([7.92822455, 6.98074183, 6.1365996, 5.38874919, 4.7273144,
                          4.14603473, 3.6361012, 3.19122823, 2.80289053, 2.46456833,
                          2.16832775, 1.9078267, 1.67708997, 1.47149048, 1.28836046,
                          1.12499443, 0.98041562, 0.86246609, 0.74137512, 0.64420517,
                          0.56007107, 0.48712409, 0.42411022, 0.36920991, 0.32113021])
    skew_GG_4 = np.array([5.70522007, 5.19393804, 4.70020995, 4.23423107, 3.80241199,
                          3.40547258, 3.04267673, 2.7131086, 2.41405058, 2.14352886,
                          1.89988063, 1.68067039, 1.48387428, 1.30726228, 1.14900746,
                          1.0069332, 0.87943423, 0.76466732, 0.66232821, 0.5702274,
                          0.48792644, 0.41489342, 0.35043604, 0.29400782, 0.24510417])
    skew_BL_4 = np.array([7.73292871, 6.80428703, 5.99305688, 5.28355563, 4.66264775,
                          4.11807141, 3.63710331, 3.21054851, 2.82965031, 2.48979637,
                          2.18650468, 1.9167441, 1.67888432, 1.46878403, 1.28456273,
                          1.12223098, 0.97965738, 0.85363241, 0.74250754, 0.64421732,
                          0.55778444, 0.48196444, 0.41588544, 0.35858012, 0.30907227])
    skew_GG_5 = np.array([5.92483714, 5.36512626, 4.83643519, 4.34455144, 3.89310056,
                          3.48033787, 3.10431515, 2.76397455, 2.45616943, 2.17859317,
                          1.92935518, 1.7056075, 1.50521957, 1.32599006, 1.16583955,
                          1.02316129, 0.89556244, 0.78151184, 0.67994803, 0.58909329,
                          0.50807331, 0.43579065, 0.37152288, 0.31471241, 0.26486283])
    skew_BL_5 = np.array([7.68413553, 6.79993205, 6.01965153, 5.3253497, 4.70489061,
                          4.15055062, 3.65539292, 3.21576027, 2.82606046, 2.48257686,
                          2.18010752, 1.91311132, 1.67797065, 1.46980395, 1.28597843,
                          1.12320827, 0.97977335, 0.85331793, 0.74229614, 0.64468134,
                          0.5588628, 0.48335674, 0.41694787, 0.3586646, 0.30772055])
    skew_GG_6 = np.array([6.15891511, 5.5465154, 4.97624331, 4.45034949, 3.96956076,
                          3.53388907, 3.14072763, 2.78888454, 2.47408017, 2.19258277,
                          1.94117173, 1.71591544, 1.51408311, 1.3332628, 1.17149729,
                          1.02748676, 0.89909759, 0.78581319, 0.68575999, 0.59769391,
                          0.52023447, 0.45201925, 0.39191142, 0.33882242, 0.29190909])
    skew_BL_6 = np.array([7.75193036, 6.84964969, 6.04465764, 5.32993556, 4.69698709,
                          4.13855959, 3.64546927, 3.21054851, 2.82569836, 2.4845567,
                          2.18260382, 1.91480956, 1.67833172, 1.46935756, 1.28529598,
                          1.12285259, 0.97976129, 0.85344799, 0.74239205, 0.64458279,
                          0.55882073, 0.48353808, 0.41753746, 0.35974383, 0.30923926])
    skew_GG_7 = np.array([6.21943578, 5.60611812, 5.03591926, 4.510632, 4.03039039,
                          3.59480927, 3.20081645, 2.84550117, 2.52601335, 2.23844223,
                          1.98062456, 1.74931391, 1.54239373, 1.35781455, 1.19345018,
                          1.04761171, 0.91791286, 0.80302526, 0.70108872, 0.61094861,
                          0.53125526, 0.46101141, 0.39921655, 0.34493501, 0.29731164])
    skew_BL_7 = np.array([7.7350363, 6.82952576, 6.03106372, 5.32445274, 4.69810973,
                          4.14281723, 3.64939869, 3.21244706, 2.82529455, 2.48351734,
                          2.18148549, 1.91436715, 1.67841389, 1.46969177, 1.28548309,
                          1.12283149, 0.97965884, 0.85337555, 0.74238914, 0.64459365,
                          0.55883652, 0.48352822, 0.41754534, 0.35981329, 0.30936906])
    skew_GG_8 = np.array([6.52705383, 5.84049599, 5.21664805, 4.65149892, 4.14135187,
                          3.68343776, 3.27190544, 2.90211162, 2.57055644, 2.27304855,
                          2.00668695, 1.76888396, 1.55710525, 1.36898515, 1.2020338,
                          1.05409561, 0.92285911, 0.80664963, 0.70365475, 0.61263627,
                          0.53231852, 0.46155411, 0.3995292, 0.34508994, 0.29740811])
    skew_BL_8 = np.array([7.74065155, 6.83765634, 6.03600024, 5.3259208, 4.69684694,
                          4.14097579, 3.64810696, 3.21230564, 2.82561677, 2.48399453,
                          2.18163626, 1.91432657, 1.67827733, 1.46960203, 1.28548309,
                          1.1228653, 0.97967181, 0.85338971, 0.74238566, 0.64457956,
                          0.55883571, 0.48353068, 0.41754552, 0.3598128, 0.30936884])
    skew_GG_9 = np.array([6.71990615, 5.99057794, 5.33402479, 4.74555576, 4.21782442,
                          3.74452106, 3.32021811, 2.93923775, 2.59831452, 2.29324446,
                          2.02114474, 1.77926525, 1.56474845, 1.37474671, 1.20640824,
                          1.05725981, 0.92502844, 0.80791846, 0.70424874, 0.61259295,
                          0.53204095, 0.46116604, 0.39904373, 0.34463208, 0.29697607])
    skew_BL_9 = np.array([7.73709557, 6.83527429, 6.03560082, 5.32675911, 4.6981838,
                          4.14170136, 3.64794946, 3.2137635, 2.8258322, 2.48402313,
                          2.18172948, 1.91434118, 1.6782484, 1.46952672, 1.28547481,
                          1.12288107, 0.97968003, 0.85347517, 0.74237942, 0.64458046,
                          0.55883825, 0.48353161, 0.41754451, 0.35981231, 0.30936913])
    skew_GG_10 = np.array([6.9507008, 6.18913983, 5.50367134, 4.88905574, 4.33728995,
                           3.84270626, 3.40112691, 3.0066214, 2.6558534, 2.34342802,
                           2.065344, 1.81812653, 1.59827517, 1.40320302, 1.23013793,
                           1.07705429, 0.94164462, 0.82187942, 0.71647795, 0.62331201,
                           0.54123401, 0.46902214, 0.4056033, 0.35002526, 0.3013423])
    skew_BL_10 = np.array([7.73840476, 6.83540451, 6.03485317, 5.32601612, 4.69862385,
                           4.14202046, 3.64811042, 3.21285433, 2.82623189, 2.48396226,
                           2.18173786, 1.91435936, 1.67825621, 1.46950687, 1.28546746,
                           1.12287902, 0.97969427, 0.85347289, 0.74237594, 0.64457956,
                           0.55884032, 0.48353284, 0.4175436, 0.35981174, 0.30936942])

    plt.plot(T, skew_GG_1, color=color(0, 10), label='N=1')
    plt.plot(T, skew_GG_2, color=color(1, 10), label='N=2')
    plt.plot(T, skew_GG_3, color=color(2, 10), label='N=3')
    plt.plot(T, skew_GG_4, color=color(3, 10), label='N=4')
    plt.plot(T, skew_GG_5, color=color(4, 10), label='N=5')
    plt.plot(T, skew_GG_6, color=color(5, 10), label='N=6')
    plt.plot(T, skew_GG_7, color=color(6, 10), label='N=7')
    plt.plot(T, skew_GG_8, color=color(7, 10), label='N=8')
    plt.plot(T, skew_GG_9, color=color(8, 10), label='N=9')
    plt.plot(T, skew_GG_10, color=color(9, 10), label='N=10')
    plt.plot(T, skew_true, color='k', label='Non-Markovian skew')
    plt.xlabel('Maturity T')
    plt.title('Skew using GG for H=-0.1')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.plot(T, skew_BL_1, color=color(0, 10), label='N=1')
    plt.plot(T, skew_BL_2, color=color(1, 10), label='N=2')
    plt.plot(T, skew_BL_3, color=color(2, 10), label='N=3')
    plt.plot(T, skew_BL_4, color=color(3, 10), label='N=4')
    plt.plot(T, skew_BL_5, color=color(4, 10), label='N=5')
    plt.plot(T, skew_BL_6, color=color(5, 10), label='N=6')
    plt.plot(T, skew_BL_7, color=color(6, 10), label='N=7')
    plt.plot(T, skew_BL_8, color=color(7, 10), label='N=8')
    plt.plot(T, skew_BL_9, color=color(8, 10), label='N=9')
    plt.plot(T, skew_BL_10, color=color(9, 10), label='N=10')
    plt.plot(T, skew_true, color='k', label='Non-Markovian skew')
    plt.xlabel('Maturity T')
    plt.title('Skew using BL for H=-0.1')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def convergence_rates_various_theorems():
    H_1 = np.linspace(-0.5, 0.5, 10001)[1:]
    H_2 = np.linspace(0, 0.5, 5001)[1:]
    plt.plot(H_1, 2.3853845 * np.sqrt(H_1 + 0.5), label='Non-geometric Gaussian Rule')
    plt.plot(H_1, 2 * np.log(1 + np.sqrt(2)) * np.sqrt(H_1 + 0.5), label='Geometric Gaussian Rule')
    plt.plot(H_2, 1.06418 * (1 + H_2 / (1.5 - H_2)) ** (-0.5) * np.sqrt(H_2), label='Bayer, Breneis, 2023')
    plt.xlabel('H')
    plt.ylabel(r'$\alpha(H)$')
    plt.legend(loc='best')
    plt.title('Exponents of the convergence rates depending on H')
    plt.tight_layout()
    plt.show()


def illustrate_bermudan_option_prices():
    """
    Bermudan option prices for the Markovian Mackevicius Paper. The parameters used are
    lambda_ = 0.3, nu = 0.3, theta = 0.02, V_0 = 0.02, rho = -0.7, S_0 = 100., T = 1., H = 0.1, K = 105., r = 0.06
    For MC estimators, the number of Monte Carlo samples used is 1_000_000, 500_000 of which are used for fitting the
    stopping rules, and 500_000 for pricing the option. The Markovian schemes use antithetic variates.
    For the Longstaff-Schwartz  linear regression, we use polynomials in S, V, and the components of V, of weighted
    degree at most 8.
    For QMC estimators, the number of QMC samples with Sobol is 2 ** 20 for the regression and 2 ** 20 for the pricing,
    and we did 25 iterations with shifted Sobol points using random shifts, taking the MC average of these 25
    iterations.
    We use 4, 16, or 256 exercise times, linearly spaced over [0, 1]. The sample paths are simulated with a number of
    time steps that is always a power of 2. In the arrays below, the prices are for increasing number of time steps in
    the simulation. For example, for 16 exercise times, these prices correspond to 16, 32, 64, 128, 256, 512, and 1024
    simulation time steps.
    The general naming convention is as follows.
    bermudan_prices_A_method_N
    where A is the number of exercise times (4, 16, or 256), method is the method of simulation (HQE, Euler, or Weak),
    and N, only for the Markovian approximations (Euler and Weak) denotes the number of dimensions for approximating
    the volatility.
    Furthermore, we have arrays for determining which weighted degree of polynomials we should use in the linear
    regression in Longstaff-Schwartz. In the end, we chose a maximal weighted degree of 8. The arrays with varying
    maximal degree are denoted by
    bermudan_prices_A_method_depending_on_d
    where A is the number of exercise times (4 or 16), and method is the method for simulation (HQE, Euler, or Weak).
    We always use 256 simulation time steps, and for the Markovian approximations, we use 3 dimensions for the
    volatility.
    All these MC estimates below of course come with MC errors. We give approximately the MC errors of the pricing
    (i.e. the second step of the Longstaff-Schwartz algorithm), in terms of a 95% confidence interval, for the prices
    below. The naming convention is
    bermudan_prices_A_MC_error
    or
    bermudan_prices_A_depending_on_d_MC_error
    where A is the number of execution times (4, 16, or 256, and 4 or 16, respectively). The MC errors are similar
    for all methods and number of simulation time steps. The 95% MC intervals are hence roughly given by, e.g.,
    bermudan_prices_4_HQE_depending_on_d +/- bermudan_prices_4_depending_on_d_MC_error.
    """

    bermudan_prices_4_HQE_depending_on_d_MC = \
        np.array([5.881312977844017, 5.968850620756691, 5.998553757861667, 6.006160182030647, 6.006383697727581,
                  6.006263235149674, 6.008178490802654, 6.008523830888406, 6.009364315695357, 6.009343528951102])
    bermudan_prices_16_HQE_depending_on_d_MC = \
        np.array([5.975877138293987, 6.111586468354794, 6.133727709676545, 6.1509455538422575, 6.154641777246265,
                  6.158014128837235, 6.158766950398943, 6.158520021365969, 6.1621527362593795, 6.159249087253855])
    bermudan_prices_4_Euler_depending_on_d_MC = \
        np.array([6.054783793139857, 6.159708441955763, 6.20228676332753, 6.216731827913274, 6.218885182596948,
                  6.218676583435156, 6.21894432367718, 6.2205907264635725, 6.217908309921049, 6.216903894341569])
    bermudan_prices_16_Euler_depending_on_d_MC = \
        np.array([6.147976009374668, 6.300566751997734, 6.3599674269505275, 6.38828958937379, 6.393769871447732,
                  6.397455068946655, 6.396819084015585, 6.398640298451329, 6.398929582688365, 6.39708370098245])
    bermudan_prices_4_Weak_depending_on_d_MC = \
        np.array([5.887155699391186, 5.989452612718753, 6.038799039128605, 6.054692018775657, 6.055529845713608,
                  6.052842941057083, 6.0551526792542925, 6.056899266851076, 6.0547788243116765, 6.054291819265685])
    bermudan_prices_16_Weak_depending_on_d_MC = \
        np.array([5.982915278022808, 6.144029283537505, 6.20662138026517, 6.231188122480796, 6.236681995867578,
                  6.239516028915048, 6.244258417204066, 6.2475150990083295, 6.243166782714627, 6.238788655289753])
    bermudan_prices_4_depending_on_d_MC_error = 0.022
    bermudan_prices_16_depending_on_d_MC_error = 0.0205

    bermudan_prices_4_HQE_QMC = \
        np.array([5.774183408437744, 5.854615351301953, 5.929369293090418, 5.9663337673351835, 5.986770239196567,
                  5.999424979520078, 6.005674224266439, 6.012039333608904, 6.016, 6.016])
    bermudan_prices_16_HQE_QMC = \
        np.array([6.0993861291263345, 6.126970973104815, 6.143719753306705, 6.151690453648474, 6.159262765596182,
                  6.163881778647127, 6.168, 6.171])
    bermudan_prices_256_HQE_QMC = \
        np.array([6.117771629784127, 6.123, 6.128, 6.130])
    bermudan_prices_4_Euler_1_QMC = \
        np.array([6.38975718366088, 6.423390029711331, 6.342266803436221, 6.234990343255016, 6.156105066476399,
                  6.11120506357235, 6.088939347127523, 6.080471413156653, 6.078006526653015, 6.074])
    bermudan_prices_4_Weak_1_QMC = \
        np.array([5.83653258591296, 5.987247751528324, 6.046365468255918, 6.065750433730277, 6.070906571147718,
                  6.072159309667814, 6.074527789330435, 6.07491176485612, 6.0767741969412, 6.075])
    bermudan_prices_16_Euler_1_QMC = \
        np.array([6.506793263507289, 6.404013763459586, 6.326878921104333, 6.28420385672242, 6.26447589920283,
                  6.254576638991984, 6.249822675780373, 6.250])
    bermudan_prices_16_Weak_1_QMC = \
        np.array([6.214071649080092, 6.237786576410557, 6.247116775253625, 6.248556966773295, 6.246794787666114,
                  6.248758449092724, 6.2478358783283605, 6.248])
    bermudan_prices_256_Euler_1_QMC = np.array([6.296, 6.284, 6.281, 6.280])
    bermudan_prices_256_Weak_1_QMC = np.array([6.277, 6.278, 6.278, 6.278])
    bermudan_prices_4_Euler_2_QMC = np.array([6.414, 6.486, 6.452, 6.355, 6.252,6.171, 6.122, 6.098, 6.086, 6.081])
    bermudan_prices_4_Weak_2_QMC = np.array([5.793, 5.952, 6.029, 6.064, 6.076, 6.076, 6.075, 6.078, 6.074, 6.074])
    bermudan_prices_16_Euler_2_QMC = np.array([6.617, 6.530, 6.428, 6.351, 6.302, 6.280, 6.268, 6.261])
    bermudan_prices_16_Weak_2_QMC = np.array([6.204, 6.237, 6.252, 6.256, 6.258, 6.257, 6.258, 6.258])
    bermudan_prices_256_Euler_2_QMC = np.array([6.338, 6.311, 6.300, 6.294])
    bermudan_prices_256_Weak_2_QMC = np.array([6.291, 6.289, 6.289, 6.288])
    bermudan_prices_4_Euler_3_QMC = np.array([6.403, 6.479, 6.481, 6.447, 6.388, 6.308, 6.225, 6.160, 6.117, 6.094])
    bermudan_prices_4_Weak_3_QMC = np.array([5.586, 5.797, 5.936, 6.015, 6.055, 6.064, 6.068, 6.074, 6.071, 6.070])
    bermudan_prices_16_Euler_3_QMC = np.array([6.645, 6.619, 6.561, 6.483, 6.402, 6.339, 6.298, 6.275])
    bermudan_prices_16_Weak_3_QMC = np.array([6.109, 6.190, 6.231, 6.245, 6.249, 6.251, 6.249, 6.249])
    bermudan_prices_256_Euler_3_QMC = np.array([6.436, 6.371, 6.332, 6.306])
    bermudan_prices_256_Weak_3_QMC = np.array([6.281, 6.281, 6.282, 6.279])
    bermudan_prices_4_Euler_4_QMC = np.array([6.403, 6.476, 6.471, 6.433, 6.396, 6.355, 6.302, 6.241, 6.182, 6.135])
    bermudan_prices_4_Weak_4_QMC = np.array([5.431, 5.619, 5.805, 5.935, 6.015, 6.052, 6.065, 6.068, 6.070, 6.070])
    bermudan_prices_16_Euler_4_QMC = np.array([6.633, 6.604, 6.565, 6.521, 6.473, 6.413, 6.355, 6.313])
    bermudan_prices_16_Weak_4_QMC = np.array([5.992, 6.114, 6.188, 6.228, 6.244, 6.247, 6.248, 6.248])
    bermudan_prices_256_Euler_4_QMC = np.array([6.505, 6.446, 6.387, 0])
    bermudan_prices_256_Weak_4_QMC = np.array([6.271, 6.275, 6.278, 0])
    bermudan_prices_4_QMC_error = 0.0025
    bermudan_prices_16_QMC_error = 0.0025
    bermudan_prices_256_QMC_error = 0.0025

    d_vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n_vec_4 = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    n_vec_16 = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    n_vec_256 = np.array([256, 512, 1024, 2048])
    plt.plot(d_vec, bermudan_prices_4_Euler_depending_on_d_MC, color='r', marker='x', label='Euler')
    plt.plot(d_vec, bermudan_prices_4_Euler_depending_on_d_MC - bermudan_prices_4_depending_on_d_MC_error, 'r--')
    plt.plot(d_vec, bermudan_prices_4_Euler_depending_on_d_MC + bermudan_prices_4_depending_on_d_MC_error, 'r--')
    plt.plot(d_vec, bermudan_prices_4_Weak_depending_on_d_MC, 'g', label='Weak')
    plt.plot(d_vec, bermudan_prices_4_Weak_depending_on_d_MC - bermudan_prices_4_depending_on_d_MC_error, 'g--')
    plt.plot(d_vec, bermudan_prices_4_Weak_depending_on_d_MC + bermudan_prices_4_depending_on_d_MC_error, 'g--')
    plt.plot(d_vec, bermudan_prices_4_HQE_depending_on_d_MC, color='b', marker='o', label='HQE')
    plt.plot(d_vec, bermudan_prices_4_HQE_depending_on_d_MC - bermudan_prices_4_depending_on_d_MC_error, 'b--')
    plt.plot(d_vec, bermudan_prices_4_HQE_depending_on_d_MC + bermudan_prices_4_depending_on_d_MC_error, 'b--')
    plt.legend(loc='best', fontsize='12')
    plt.xlabel('Weighted maximal degree d of polynomials', fontsize='12')
    plt.title('Prices of Bermudan put option with 4 execution times', fontsize='12')
    plt.tight_layout()
    plt.show()

    plt.plot(d_vec, bermudan_prices_16_Euler_depending_on_d_MC, color='r', marker='x', label='Euler')
    plt.plot(d_vec, bermudan_prices_16_Euler_depending_on_d_MC - bermudan_prices_16_depending_on_d_MC_error, 'r--')
    plt.plot(d_vec, bermudan_prices_16_Euler_depending_on_d_MC + bermudan_prices_16_depending_on_d_MC_error, 'r--')
    plt.plot(d_vec, bermudan_prices_16_Weak_depending_on_d_MC, 'g', label='Weak')
    plt.plot(d_vec, bermudan_prices_16_Weak_depending_on_d_MC - bermudan_prices_16_depending_on_d_MC_error, 'g--')
    plt.plot(d_vec, bermudan_prices_16_Weak_depending_on_d_MC + bermudan_prices_16_depending_on_d_MC_error, 'g--')
    plt.plot(d_vec, bermudan_prices_16_HQE_depending_on_d_MC, color='b', marker='o', label='HQE')
    plt.plot(d_vec, bermudan_prices_16_HQE_depending_on_d_MC - bermudan_prices_16_depending_on_d_MC_error, 'b--')
    plt.plot(d_vec, bermudan_prices_16_HQE_depending_on_d_MC + bermudan_prices_16_depending_on_d_MC_error, 'b--')
    plt.legend(loc='best', fontsize='12')
    plt.xlabel('Weighted maximal degree d of polynomials', fontsize='12')
    plt.title('Prices of Bermudan put option with 16 execution times', fontsize='12')
    plt.tight_layout()
    plt.show()

    plt.plot(n_vec_4, bermudan_prices_4_Euler_1_QMC, color='r', marker='x', label='Euler, N=1')
    plt.plot(n_vec_4, bermudan_prices_4_Euler_2_QMC, color='r', marker='x', linestyle='dotted', label='Euler, N=2')
    plt.plot(n_vec_4, bermudan_prices_4_Euler_3_QMC, color='r', marker='x', linestyle='dashed', label='Euler, N=3')
    # plt.plot(n_vec_4, bermudan_prices_4_Euler_4_QMC, color='r', marker='x', linestyle='dashdot', label='Euler, N=4')
    plt.plot(n_vec_4, bermudan_prices_4_Weak_1_QMC, color='g', label='Weak, N=1')
    plt.plot(n_vec_4, bermudan_prices_4_Weak_2_QMC, color='g', linestyle='dotted', label='Weak, N=2')
    plt.plot(n_vec_4, bermudan_prices_4_Weak_3_QMC, color='g', linestyle='dashed', label='Weak, N=3')
    # plt.plot(n_vec_4, bermudan_prices_4_Weak_4_QMC, color='g', linestyle='dashdot', label='Weak, N=4')
    plt.plot(n_vec_4, bermudan_prices_4_HQE_QMC, color='b', marker='o', label='HQE')
    # plt.plot(n_vec_4, bermudan_prices_4_HQE_QMC - bermudan_prices_4_QMC_error, 'b--')
    # plt.plot(n_vec_4, bermudan_prices_4_HQE_QMC + bermudan_prices_4_QMC_error, 'b--')
    plt.xscale('log')
    plt.legend(loc='best', fontsize='12')
    plt.xlabel('Number of simulation time steps', fontsize='12')
    plt.title('Prices of Bermudan put option with 4 execution times', fontsize='12')
    plt.tight_layout()
    plt.show()

    plt.plot(n_vec_16, bermudan_prices_16_Euler_1_QMC, color='r', marker='x', label='Euler, N=1')
    plt.plot(n_vec_16, bermudan_prices_16_Euler_2_QMC, color='r', marker='x', linestyle='dotted', label='Euler, N=2')
    plt.plot(n_vec_16, bermudan_prices_16_Euler_3_QMC, color='r', marker='x', linestyle='dashed', label='Euler, N=3')
    # plt.plot(n_vec_16, bermudan_prices_16_Euler_4_QMC, color='r', marker='x', linestyle='dashdot', label='Euler, N=4')
    plt.plot(n_vec_16, bermudan_prices_16_Weak_1_QMC, color='g', label='Weak, N=1')
    plt.plot(n_vec_16, bermudan_prices_16_Weak_2_QMC, color='g', linestyle='dotted', label='Weak, N=2')
    plt.plot(n_vec_16, bermudan_prices_16_Weak_3_QMC, color='g', linestyle='dashed', label='Weak, N=3')
    # plt.plot(n_vec_16, bermudan_prices_16_Weak_4_QMC, color='g', linestyle='dashdot', label='Weak, N=4')
    plt.plot(n_vec_16, bermudan_prices_16_HQE_QMC, color='b', marker='o', label='HQE')
    # plt.plot(n_vec_16, bermudan_prices_16_HQE_QMC - bermudan_prices_16_QMC_error, 'b--')
    # plt.plot(n_vec_16, bermudan_prices_16_HQE_QMC + bermudan_prices_16_QMC_error, 'b--')
    plt.xscale('log')
    plt.legend(loc='best', fontsize='12')
    plt.xlabel('Number of simulation time steps', fontsize='12')
    plt.title('Prices of Bermudan put option with 16 execution times', fontsize='12')
    plt.tight_layout()
    plt.show()

    plt.plot(n_vec_256, bermudan_prices_256_Euler_1_QMC, color='r', marker='x', label='Euler, N=1')
    plt.plot(n_vec_256, bermudan_prices_256_Euler_2_QMC, color='r', marker='x', linestyle='dotted', label='Euler, N=2')
    plt.plot(n_vec_256, bermudan_prices_256_Euler_3_QMC, color='r', marker='x', linestyle='dashed', label='Euler, N=3')
    plt.plot(n_vec_256, bermudan_prices_256_Euler_4_QMC, color='r', marker='x', linestyle='dashdot', label='Euler, N=4')
    plt.plot(n_vec_256, bermudan_prices_256_Weak_1_QMC, color='g', label='Weak, N=1')
    plt.plot(n_vec_256, bermudan_prices_256_Weak_2_QMC, color='g', linestyle='dotted', label='Weak, N=2')
    plt.plot(n_vec_256, bermudan_prices_256_Weak_3_QMC, color='g', linestyle='dashed', label='Weak, N=3')
    plt.plot(n_vec_256, bermudan_prices_256_Weak_4_QMC, color='g', linestyle='dashdot', label='Weak, N=4')
    plt.plot(n_vec_256, bermudan_prices_256_HQE_QMC, color='b', marker='o', label='HQE')
    # plt.plot(n_vec_256, bermudan_prices_256_HQE_QMC - bermudan_prices_256_QMC_error, 'g--')
    # plt.plot(n_vec_256, bermudan_prices_256_HQE_QMC + bermudan_prices_256_QMC_error, 'g--')
    plt.xscale('log')
    plt.legend(loc='best', fontsize='12')
    plt.xlabel('Number of simulation time steps', fontsize='12')
    plt.title('Prices of Bermudan put option with 256 execution times', fontsize='12')
    plt.tight_layout()
    plt.show()

    plt.plot(n_vec_4, bermudan_prices_4_Euler_3_QMC, color='r', marker='x', label='Euler, 4 ex. times')
    plt.plot(n_vec_16, bermudan_prices_16_Euler_3_QMC, color='r', marker='x', linestyle='dotted',
             label='Euler, 16 ex. times')
    plt.plot(n_vec_256, bermudan_prices_256_Euler_3_QMC, color='r', marker='x', linestyle='dashed',
             label='Euler, 256 ex. times')
    plt.plot(n_vec_4, bermudan_prices_4_Weak_3_QMC, color='g', label='Weak, 4 ex. times')
    plt.plot(n_vec_16, bermudan_prices_16_Weak_3_QMC, color='g', linestyle='dotted', label='Weak, 16 ex. times')
    plt.plot(n_vec_256, bermudan_prices_256_Weak_3_QMC, color='g', linestyle='dashed', label='Weak, 256 ex. times')
    plt.plot(n_vec_4, bermudan_prices_4_HQE_QMC, color='b', marker='o', label='HQE, 4 ex. times')
    # plt.plot(n_vec_4, bermudan_prices_4_HQE_QMC - bermudan_prices_4_QMC_error, color='g', linestyle='dashdot')
    # plt.plot(n_vec_4, bermudan_prices_4_HQE_QMC + bermudan_prices_4_QMC_error, color='g', linestyle='dashdot')
    plt.plot(n_vec_16, bermudan_prices_16_HQE_QMC, color='b', marker='o', linestyle='dotted', label='HQE, 16 ex. times')
    plt.plot(n_vec_256, bermudan_prices_256_HQE_QMC, color='b', marker='o', linestyle='dashed',
             label='HQE, 256 ex. times')
    plt.xscale('log')
    plt.legend(loc='best', fontsize='12')
    plt.xlabel('Number of simulation time steps', fontsize='12')
    plt.title('Prices of Bermudan put options', fontsize='12')
    plt.tight_layout()
    plt.show()


def plot_for_L1_error_bound_paper_comparing_OL1_and_OL2():
    N = np.arange(1, 11)
    H = np.array([-0.2, -0.1, 0.001, 0.1, 0.2])
    rel_tol = 1.5e-05
    errors_OL2 = np.empty((5, 10))
    errors_OL1 = np.empty((5, 10))
    errors_OL1[0, :] = np.array([0.16578209054795429, 0.06674748107848859, 0.03395972637809162, 0.01907663728373314,
                                 0.01129600187865071, 0.007125028168034053, 0.004617895979875806, 0.003063735391021798,
                                 0.0020819982454842244, 0.0012704762273173678])
    errors_OL1[1, :] = np.array([0.1232849144263059, 0.04296636130089774, 0.01935154492620456, 0.009751101040076465,
                                 0.005167505009098943, 0.0030263745794344827, 0.0018066519747875064,
                                 0.0011114762832229012,
                                 0.0005962176214555475, 0.0003536868542463575])
    errors_OL1[2, :] = np.array([0.09281444748013044, 0.02833225773066702, 0.011332015461682747, 0.005151131573622105,
                                 0.0025580577463198547, 0.001349644650250876, 0.0006363686097173661,
                                 0.0004310694880891412,
                                 0.0002084546602560261, 0.00010723095194550676])
    errors_OL1[3, :] = np.array([0.06961192417520104, 0.018788772116170027, 0.006715709859832943, 0.0027771268847164485,
                                 0.0012730871146334891, 0.0006245324827054702, 0.0003246310191285324,
                                 0.00017687107710697176, 7.376435847385586e-05, 3.8839031612146845e-05])
    errors_OL1[4, :] = np.array(
        [0.049855557274949586, 0.011917024073545359, 0.003815642237603592, 0.0014428499357432695,
         0.000610956126995606, 0.000234586481157565, 0.00013724270120548285, 5.270258132911103e-05,
         2.364679243620818e-05, 1.1357094390006162e-05])
    errors_OL2[2, :] = np.array([0.07678411911770376, 0.06540793660346071, 0.05834883208127644, 0.05317043318153089,
                                 0.048735334499656835, 0.04512928469724931, 0.04275685053861919, 0.04022168863463462,
                                 0.03797532169027432, 0.035960481790459194])
    errors_OL2[3, :] = np.array([0.008972384863547072, 0.006060526330399336, 0.005250579019756992, 0.004119083031922951,
                                 0.00312741935266171, 0.002343247096380104, 0.0017458818775902222, 0.001298457021612445,
                                 0.0009661688536627486, 0.0007204192911469224])
    errors_OL2[4, :] = np.array(
        [0.007776048320412874, 0.004642384659836375, 0.0025530608290878633, 0.0013939902812472433,
         0.0007631860432554396, 0.0004226475414530141, 0.00023763575556596018,
         0.0001354774416875561, 7.794484224425964e-05, 4.494984305874949e-05])

    for i in range(len(H)):
        plt.loglog(N, errors_OL1[i, :], color=color(i, len(H)), label=f'H={H[i]}')
        if H[i] > 0:
            plt.loglog(N, errors_OL2[i, :], '--', color=color(i, len(H)))
    # plt.loglog(N, 2 * rel_tol * np.ones_like(N), 'k-')
    plt.xlabel('Number of nodes N')
    plt.legend(loc='best')
    plt.title('Maximal relative errors of implied volatility smiles')
    plt.tight_layout()
    plt.show()


def plot_for_simulation_paper_smile_errors():
    n_smile = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    N_smile = np.array([2, 3])
    H_smile = np.array([0.1, -0.2])
    HQE_smile = np.empty((3, len(H_smile), len(n_smile)))  # total, MC, excess
    Euler_smile = np.empty((4, len(H_smile), len(N_smile), len(n_smile)))  # total, discr, MC, excess
    Weak_smile = np.empty_like(Euler_smile)
    Markov_smile = np.empty((len(H_smile), len(N_smile)))
    nodes_smile = np.empty((len(H_smile), len(N_smile)))
    HQE_smile[0, 0, :] = np.array([10.85, 11.06, 8.124, 4.591, 2.24, 1.158, 0.7272, 0.4846, 0.3124, 0.1912, 0.1588])
    HQE_smile[1, 0, :] = np.array([0.0004697, 0.001584, 0.003557, 0.007299, 0.008072, 0.01366, 0.01461, 0.01601,
                                   0.01977, 0.02333, 0.01790])
    HQE_smile[2, 0, :] = np.array([10.85, 11.06, 8.121, 4.584, 2.232, 1.147, 0.7137, 0.4716, 0.2946, 0.1734, 0.1409])

    Markov_smile[0, 0] = 0.0001323
    nodes_smile[0, 0] = 8.7171
    Euler_smile[0, 0, 0, :] = np.array([17.95, 14.86, 13.92, 12.14, 9.671, 6.775, 4.162, 2.255, 1.113, 0.5137, 0.222])
    Euler_smile[1, 0, 0, :] = np.array([17.94, 14.86, 13.93, 12.15, 9.678, 6.782, 4.169, 2.261, 1.119, 0.5204, 0.229])
    Euler_smile[2, 0, 0, :] = np.array([0.0001606, 0.0004813, 0.009113, 0.007643, 0.009972, 0.01096, 0.01916, 0.01753,
                                        0.01984, 0.01788, 0.0183])
    Euler_smile[3, 0, 0, :] = np.array([17.94, 14.86, 13.92, 12.14, 9.668, 6.774, 4.152, 2.251, 1.104, 0.5046, 0.214])
    Weak_smile[0, 0, 0, :] = np.array([25.1, 15.94, 7.161, 2.397, 0.6655, 0.1778, 0.05295, 0.0199, 0.0199, 0.01589,
                                       0.013])
    Weak_smile[1, 0, 0, :] = np.array([25.09, 15.93, 7.151, 2.384, 0.6523, 0.1647, 0.04005, 0.007346, 0.01603,
                                       0.004907, 0.001])
    Weak_smile[2, 0, 0, :] = np.array([0.0002869, 0.0009064, 0.01148, 0.005592, 0.01006, 0.0142, 0.01477, 0.01766,
                                       0.01839, 0.02026, 0.0179])
    Weak_smile[3, 0, 0, :] = np.array([25.09, 15.93, 7.147, 2.379, 0.6441, 0.1545, 0.02927, 0.0, 0.00263, 0.0, 0.0])

    Markov_smile[0, 1] = 0.0001061
    nodes_smile[0, 1] = 46.831
    Euler_smile[0, 0, 1, :] = np.array([17.95, 14.64, 13.71, 11.98, 10.4, 8.919, 7.265, 5.358, 3.495, 2.031, 1.074])
    Euler_smile[1, 0, 1, :] = np.array([17.95, 14.66, 13.72, 11.99, 10.41, 8.931, 7.277, 5.369, 3.506, 2.042, 1.084])
    Euler_smile[2, 0, 1, :] = np.array([0.000182, 0.0003747, 0.007863, 0.00598, 0.008593, 0.01202, 0.01517, 0.01802,
                                        0.01926, 0.0177, 0.0220])
    Euler_smile[3, 0, 1, :] = np.array([17.95, 14.66, 13.71, 11.99, 10.4, 8.921, 7.265, 5.354, 3.491, 2.029, 1.065])
    Weak_smile[0, 0, 1, :] = np.array([32.36, 25.5, 16.58, 8.813, 3.544, 1.068, 0.2818, 0.06156, 0.02817, 0.01499,
                                       0.012])
    Weak_smile[1, 0, 1, :] = np.array([32.37, 25.5, 16.59, 8.816, 3.545, 1.066, 0.2799, 0.05924, 0.02336, 0.008746,
                                       0.004])
    Weak_smile[2, 0, 1, :] = np.array([0.0002066, 0.0005943, 0.01481, 0.004585, 0.007014, 0.01097, 0.01412, 0.01592,
                                       0.01547, 0.01598, 0.0194])
    Weak_smile[3, 0, 1, :] = np.array([32.37, 25.5, 16.58, 8.812, 3.538, 1.057, 0.2678, 0.04658, 0.01061, 0.0, 0.0])

    Markov_smile[1, 0] = 0.000649
    nodes_smile[1, 0] = 60.452
    Euler_smile[0, 1, 0, :] = np.array([18.44, 18.10, 26.52, 35.55, 45.07, 53.27, 56.99, 54.27, 45.98, 35.11, 24.50])
    Euler_smile[1, 1, 0, :] = np.array([18.41, 18.03, 26.44, 35.47, 44.98, 53.17, 56.88, 54.17, 45.89, 35.02, 24.42])
    Euler_smile[2, 1, 0, :] = np.array([0.0001594, 0.000445, 0.004552, 0.006467, 0.01454, 0.01542, 0.01483, 0.02012,
                                        0.02537, 0.02464, 0.0279])
    Euler_smile[3, 1, 0, :] = np.array([18.41, 18.03, 26.43, 35.46, 44.96, 53.15, 56.87, 54.15, 45.86, 34.99, 24.39])
    Weak_smile[0, 1, 0, :] = np.array([37.52, 36.86, 33.41, 27.02, 17.58, 7.885, 2.636, 0.7081, 0.1587, 0.06635, 0.064])
    Weak_smile[1, 1, 0, :] = np.array([37.50, 36.84, 33.39, 27.00, 17.55, 7.862, 2.636, 0.7224, 0.1714, 0.03964, 0.012])
    Weak_smile[2, 1, 0, :] = np.array([0.0002219, 0.0005522, 0.01815, 0.004001, 0.005323, 0.007084, 0.0122, 0.01457,
                                       0.02133, 0.02897, 0.0304])
    Weak_smile[3, 1, 0, :] = np.array([37.50, 36.84, 33.38, 26.99, 17.55, 7.857, 2.626, 0.7089, 0.1569, 0.01842, 0.0])

    Markov_smile[1, 1] = 0.0000597
    nodes_smile[1, 1] = 681.37
    Euler_smile[0, 1, 1, :] = np.array([18.44, 17.94, 25.00, 31.27, 37.87, 45.33, 54.13, 63.63, 71.63, 75.51, 73.81])
    Euler_smile[1, 1, 1, :] = np.array([18.44, 17.93, 25.00, 31.27, 37.86, 45.32, 54.12, 63.62, 71.62, 75.50, 73.80])
    Euler_smile[2, 1, 1, :] = np.array([0.000182, 0.0005123, 0.006125, 0.006301, 0.009887, 0.01359, 0.02281, 0.02178,
                                        0.02452, 0.02768, 0.0253])
    Euler_smile[3, 1, 1, :] = np.array([18.44, 17.93, 24.99, 31.26, 37.85, 45.30, 54.09, 63.59, 71.59, 75.46, 73.77])
    Weak_smile[0, 1, 1, :] = np.array([38.74, 40.39, 40.19, 38.70, 35.64, 30.36, 22.71, 13.83, 5.945, 1.859, 0.478])
    Weak_smile[1, 1, 1, :] = np.array([38.74, 40.39, 40.18, 38.69, 35.63, 30.36, 22.71, 13.83, 5.944, 1.860, 0.479])
    Weak_smile[2, 1, 1, :] = np.array([0.0001541, 0.0004758, 0.02254, 0.005561, 0.005572, 0.006918, 0.007507, 0.00729,
                                       0.01039, 0.0138, 0.0229])
    Weak_smile[3, 1, 1, :] = np.array([38.74, 40.39, 40.17, 38.69, 35.63, 30.35, 22.71, 13.83, 5.935, 1.850, 0.460])

    HQE_smile, Euler_smile, Weak_smile = 0.01 * HQE_smile, 0.01 * Euler_smile, 0.01 * Weak_smile

    n_surface = np.array([16, 32, 64, 128, 256, 512, 1024])
    N_surface = np.array([2, 3])
    H_surface = 0.1
    HQE_surface = np.empty((3, len(n_surface)))  # total, MC, excess
    Euler_surface = np.empty((4, len(N_surface), len(n_surface)))  # total, discr, MC, excess
    Weak_surface = np.empty_like(Euler_surface)
    Markov_surface = np.empty(len(N_surface))
    nodes_surface = np.empty(len(N_surface))

    HQE_surface[0, :] = np.array([8.631, 7.914, 5.467, 3.036, 1.565, 0.935, 0.58])
    HQE_surface[1, :] = np.array([0.0114, 0.0148, 0.0174, 0.0189, 0.0251, 0.0230, 0.025])
    HQE_surface[2, :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    Markov_surface[0] = 0.0163
    nodes_surface[0] = 34.87
    Euler_surface[0, 0, :] = np.array([26.06, 24.27, 16.57, 9.440, 4.593, 2.240, 1.250])
    Euler_surface[1, 0, :] = np.array([27.86, 26.04, 18.23, 11.00, 6.088, 2.064, 1.437])
    Euler_surface[2, 0, :] = np.array([0.01605, 0.01226, 0.01934, 0.02007, 0.01791, 0.01815, 0.0226])
    Euler_surface[3, 0, :] = np.array([27.86, 26.04, 18.23, 10.99, 6.078, 3.056, 1.427])
    Weak_surface[0, 0, :] = np.array([10.21, 4.686, 2.183, 1.625, 1.632, 1.639, 1.643])
    Weak_surface[1, 0, :] = np.array([9.958, 4.160, 1.390, 0.3897, 0.1050, 0.03640, 0.027])
    Weak_surface[2, 0, :] = np.array([0.009144, 0.01287, 0.0168, 0.02008, 0.02257, 0.01981, 0.0204])
    Weak_surface[3, 0, :] = np.array([9.957, 4.159, 1.383, 0.382, 0.093, 0.027, 0.011])

    Markov_surface[1] = 0.001867
    nodes_surface[1] = 118.01
    Euler_surface[0, 1, :] = np.array([26.06, 24.31, 17.92, 12.81, 8.801, 5.462, 3.036])
    Euler_surface[1, 1, :] = np.array([26.13, 24.37, 17.98, 12.87, 8.856, 5.515, 3.078])
    Euler_surface[2, 1, :] = np.array([0.0244, 0.0178, 0.0151, 0.0201, 0.0187, 0.0255, 0.0231])
    Euler_surface[3, 1, :] = np.array([26.13, 24.37, 17.97, 12.86, 8.843, 5.501, 3.068])
    Weak_surface[0, 1, :] = np.array([16.35, 8.689, 3.656, 1.261, 0.428, 0.200, 0.1899])
    Weak_surface[1, 1, :] = np.array([16.25, 8.571, 3.528, 1.134, 0.306, 0.067, 0.02565])
    Weak_surface[2, 1, :] = np.array([0.0064, 0.0120, 0.0202, 0.0218, 0.0193, 0.0203, 0.0226])
    Weak_surface[3, 1, :] = np.array([16.25, 8.569, 3.525, 1.127, 0.295, 0.057, 0.0133])

    HQE_surface, Euler_surface, Weak_surface = 0.01 * HQE_surface, 0.01 * Euler_surface, 0.01 * Weak_surface

    n_asian = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    N_asian = np.array([2, 3])
    H_asian = 0.1
    HQE_asian = np.empty((3, len(n_asian)))  # total, MC, excess
    Euler_asian = np.empty((4, len(N_asian), len(n_asian)))  # total, discr, MC, excess
    Weak_asian = np.empty_like(Euler_asian)
    Markov_asian = np.empty(len(N_asian))
    nodes_asian = np.empty(len(N_asian))

    HQE_asian[0, :] = np.array([38.58, 17.95, 13.40, 9.606, 5.722, 2.858, 1.387, 0.750, 0.458, 0.2825, 0.1962])
    HQE_asian[1, :] = np.array([0.0015, 0.0023, 0.0050, 0.0107, 0.0175, 0.0227, 0.0331, 0.0357, 0.0378, 0.03944,
                                0.03588])

    Markov_asian[0] = 0.005292
    nodes_asian[0] = 8.7171
    Euler_asian[0, 0, :] = np.array([14.44, 40.72, 44.82, 39.22, 30.29, 20.32, 11.79, 5.916, 2.585, 0.9034, 0.07106])
    Euler_asian[1, 0, :] = np.array([14.39, 41.47, 45.59, 39.96, 30.98, 20.96, 12.38, 6.480, 3.131, 1.440, 0.6035])
    Euler_asian[2, 0, :] = np.array([0.0002647, 0.0004465, 0.003508, 0.008856, 0.01676, 0.02242, 0.02278, 0.03726,
                                     0.02932, 0.03022, 0.02423])
    Euler_asian[3, 0, :] = np.array([14.39, 41.47, 45.58, 39.95, 30.96, 20.94, 12.36, 6.440, 3.101, 1.410, 0.5791])
    Weak_asian[0, 0, :] = np.array([30.12, 13.98, 6.958, 3.374, 1.270, 0.4938, 0.3819, 0.5134, 0.5259, 0.5190, 0.521])
    Weak_asian[1, 0, :] = np.array([29.81, 13.84, 6.827, 3.924, 1.809, 0.5632, 0.1579, 0.02332, 0.004048, 0.01027,
                                    0.008])
    Weak_asian[2, 0, :] = np.array([0.0004367, 0.001021, 0.005349, 0.01259, 0.01438, 0.01903, 0.02597, 0.03436, 0.02909,
                                    0.03447, 0.03])
    Weak_asian[3, 0, :] = np.array([29.81, 13.84, 6.825, 3.911, 1.795, 0.5440, 0.1319, 0.009505, 0.000, 0.000, 0.])

    Markov_asian[1] = 0.0001233
    nodes_asian[1] = 46.83
    Euler_asian[0, 1, :] = np.array([14.44, 40.66, 44.57, 39.47, 33.05, 27.23, 21.43, 15.34, 9.752, 5.576, 2.872])
    Euler_asian[1, 1, :] = np.array([14.44, 40.65, 44.55, 39.45, 33.03, 27.21, 21.41, 15.44, 9.739, 5.563, 2.860])
    Euler_asian[2, 1, :] = np.array([0.0003765, 0.0005353, 0.003264, 0.009283, 0.01854, 0.02264, 0.02179, 0.03309,
                                     0.03353, 0.03223, 0.03112])
    Euler_asian[3, 1, :] = np.array([14.44, 40.64, 44.55, 39.44, 33.00, 27.18, 21.39, 15.29, 9.702, 5.529, 2.828])
    Weak_asian[0, 1, :] = np.array([33.64, 19.22, 12.44, 7.092, 5.402, 2.616, 0.8532, 0.2004, 0.06884, 0.009356, 0.014])
    Weak_asian[1, 1, :] = np.array([33.65, 19.23, 12.45, 7.100, 5.389, 2.604, 0.8408, 0.1881, 0.05651, 0.01915, 0.014])
    Weak_asian[2, 1, :] = np.array([0.0004728, 0.001117, 0.005183, 0.01191, 0.01615, 0.01926, 0.02554, 0.03478, 0.02860,
                                    0.03551, 0.032])
    Weak_asian[3, 1, :] = np.array([33.64, 19.23, 12.45, 7.096, 5.372, 2.584, 0.8150, 0.1532, 0.02789, 0.002235, 0.0])

    '''
    Markov_asian[0] = 0.003195
    nodes_asian[0] = 12.328
    Euler_asian[0, 0, :] = np.array([14.44, 40.76, 44.80, 39.67, 31.92, 22.91, 14.35, 7.845, 3.816, 1.609])
    Euler_asian[1, 0, :] = np.array([14.42, 41.21, 45.27, 40.12, 32.34, 23.30, 14.72, 8.190, 4.149, 1.935])
    Euler_asian[2, 0, :] = np.array([0.0003, 0.0004, 0.0031, 0.0066, 0.0141, 0.0142, 0.0201, 0.0316, 0.0270, 0.0314])
    Euler_asian[3, 0, :] = np.array([14.42, 41.21, 45.26, 40.11, 32.32, 23.28, 14.70, 8.156, 4.121, 1.903])
    Weak_asian[0, 0, :] = np.array([30.86, 15.16, 7.970, 4.318, 2.040, 0.473, 0.215, 0.251, 0.321, 0.333])
    Weak_asian[1, 0, :] = np.array([30.69, 15.08, 7.909, 4.653, 2.367, 0.782, 0.215, 0.069, 0.011, 0.014])
    Weak_asian[2, 0, :] = np.array([0.0007, 0.0011, 0.0048, 0.0104, 0.0126, 0.0189, 0.0302, 0.0277, 0.0331, 0.0255])
    Weak_asian[3, 0, :] = np.array([30.68, 15.08, 7.907, 4.642, 2.354, 0.763, 0.185, 0.041, 0.000, 0.000])

    Markov_asian[1] = 0.000026
    nodes_asian[1] = 59.003
    Euler_asian[0, 1, :] = np.array([14.44, 40.67, 44.56, 39.35, 32.81, 27.12, 21.84, 16.24, 10.76, 6.394])
    Euler_asian[1, 1, :] = np.array([14.44, 40.67, 44.56, 39.36, 32.82, 27.12, 21.84, 16.24, 10.76, 6.397])
    Euler_asian[2, 1, :] = np.array([0.0003, 0.0004, 0.0026, 0.0060, 0.0127, 0.0139, 0.0175, 0.0284, 0.0333, 0.0299])
    Euler_asian[3, 1, :] = np.array([14.44, 40.67, 44.56, 39.35, 32.80, 27.11, 21.82, 16.21, 10.73, 6.365])
    Weak_asian[0, 1, :] = np.array([33.93, 19.76, 13.20, 7.793, 5.823, 3.115, 1.066, 0.262, 0.090, 0.027])
    Weak_asian[1, 1, :] = np.array([33.93, 19.76, 13.20, 7.794, 5.826, 3.118, 1.068, 0.264, 0.092, 0.025])
    Weak_asian[2, 1, :] = np.array([0.0006, 0.0018, 0.0039, 0.0082, 0.0082, 0.0180, 0.0223, 0.0318, 0.0312, 0.0306])
    Weak_asian[3, 1, :] = np.array([33.93, 19.76, 13.20, 7.791, 5.817, 3.099, 1.045, 0.233, 0.061, 0.008])
    '''

    HQE_asian, Euler_asian, Weak_asian = 0.01 * HQE_asian, 0.01 * Euler_asian, 0.01 * Weak_asian

    for i in range(len(H_smile)):
        for j in range(len(N_smile)):
            plt.loglog(n_smile, Euler_smile[0, i, j, :], color='r', marker='x', label='Euler')
            plt.loglog(n_smile, Euler_smile[0, i, j, :] - Euler_smile[2, i, j, :], 'r--')
            plt.loglog(n_smile, Euler_smile[0, i, j, :] + Euler_smile[2, i, j, :], 'r--')
            plt.loglog(n_smile, Weak_smile[0, i, j, :], 'g-', label='Weak')
            plt.loglog(n_smile, Weak_smile[0, i, j, :] - Weak_smile[2, i, j, :], 'g--')
            plt.loglog(n_smile, Weak_smile[0, i, j, :] + Weak_smile[2, i, j, :], 'g--')
            if H_smile[i] > 0:
                plt.loglog(n_smile, HQE_smile[0, i, :], color='b', marker='o', label='HQE')
                plt.loglog(n_smile, HQE_smile[0, i, :] - HQE_smile[1, i, :], 'b--')
                plt.loglog(n_smile, HQE_smile[0, i, :] + HQE_smile[1, i, :], 'b--')
            plt.loglog(n_smile, Markov_smile[i, j] * np.ones_like(n_smile), 'k-', label='Markov error')
            plt.loglog(nodes_smile[i, j] * np.ones(2), np.array([0, 10]), 'k--', label='Largest node')
            if i == 0:
                plt.loglog(n_smile, HQE_smile[0, i, 9] * 512 * n_smile ** (-1.), 'k:', label='Rate 1')
                plt.loglog(n_smile, Weak_smile[0, i, j, 5] * 32 ** 2 * n_smile ** (-2.), 'k-.', label='Rate 2')
                plt.ylim(0.00005, 0.4)
            if i == 1 and j == 0:
                plt.loglog(n_smile, Euler_smile[0, i, j, 3] * 8 * n_smile ** (-1.), 'k:', label='Rate 1')
                plt.loglog(n_smile, Weak_smile[0, i, j, 6] * 64 ** 2 * n_smile ** (-2.), 'k-.', label='Rate 2')
                plt.ylim(0.0002, 1.)
            if i == 1 and j == 1:
                plt.loglog(n_smile, Euler_smile[0, i, j, 5] * 32 * n_smile ** (-1.), 'k:', label='Rate 1')
                plt.loglog(n_smile, Weak_smile[0, i, j, 9] * 512 ** 2 * n_smile ** (-2.), 'k-.', label='Rate 2')
                plt.ylim(0.00003, 1.2)
            plt.xlabel('Number of time steps', fontsize='12')
            plt.title(f'Maximal relative errors of IV smiles with H={H_smile[i]} and N={N_smile[j]}', fontsize='12')
            plt.legend(loc='best', fontsize='12')
            plt.tight_layout()
            plt.show()

    for j in range(len(N_surface)):
        plt.loglog(n_surface, Euler_surface[0, j, :], color='r', marker='x', label='Euler')
        plt.loglog(n_surface, Euler_surface[0, j, :] - Euler_surface[2, j, :], 'r--')
        plt.loglog(n_surface, Euler_surface[0, j, :] + Euler_surface[2, j, :], 'r--')
        plt.loglog(n_surface, Weak_surface[0, j, :], 'g-', label='Weak')
        plt.loglog(n_surface, Weak_surface[0, j, :] - Weak_surface[2, j, :], 'g--')
        plt.loglog(n_surface, Weak_surface[0, j, :] + Weak_surface[2, j, :], 'g--')
        plt.loglog(n_surface, HQE_surface[0, :], color='b', marker='o', label='HQE')
        plt.loglog(n_surface, HQE_surface[0, :] - HQE_surface[1, :], 'b--')
        plt.loglog(n_surface, HQE_surface[0, :] + HQE_surface[1, :], 'b--')
        plt.loglog(n_surface, Markov_surface[j] * np.ones_like(n_surface), 'k-', label='Markov error')
        plt.loglog(nodes_surface[j] * np.ones(2), np.array([0, 10]), 'k--', label='Largest node')
        plt.loglog(n_surface, HQE_surface[0, 5] * 512 * n_surface ** (-1.), 'k:', label='Rate 1')
        if j == 0:
            plt.loglog(n_surface, Weak_surface[0, j, 2] * 64 ** 2 * n_surface ** (-2.), 'k-.', label='Rate 2')
            plt.ylim(0.004, 0.4)
        if j == 1:
            plt.loglog(n_surface, Weak_surface[0, j, 3] * 128 ** 2 * n_surface ** (-2.), 'k-.', label='Rate 2')
            plt.ylim(0.001, 0.4)
        plt.xlabel('Number of time steps', fontsize='12')
        plt.title(f'Maximal relative errors of IV surfaces with H={H_surface} and N={N_surface[j]}', fontsize='12')
        plt.legend(loc='best', fontsize='12')
        plt.tight_layout()
        plt.show()

    for j in range(len(N_asian)):
        plt.loglog(n_asian, Euler_asian[0, j, :], color='r', marker='x', label='Euler')
        plt.loglog(n_asian, Euler_asian[0, j, :] - Euler_asian[2, j, :], 'r--')
        plt.loglog(n_asian, Euler_asian[0, j, :] + Euler_asian[2, j, :], 'r--')
        plt.loglog(n_asian, Weak_asian[0, j, :], 'g-', label='Weak')
        plt.loglog(n_asian, Weak_asian[0, j, :] - Weak_asian[2, j, :], 'g--')
        plt.loglog(n_asian, Weak_asian[0, j, :] + Weak_asian[2, j, :], 'g--')
        plt.loglog(n_asian, HQE_asian[0, :], color='b', marker='o', label='HQE')
        plt.loglog(n_asian, HQE_asian[0, :] - HQE_asian[1, :], 'b--')
        plt.loglog(n_asian, HQE_asian[0, :] + HQE_asian[1, :], 'b--')
        plt.loglog(n_asian, Markov_asian[j] * np.ones_like(n_asian), 'k-', label='Markov error')
        plt.loglog(nodes_asian[j] * np.ones(2), np.array([0, 10]), 'k--', label='Largest node')
        plt.loglog(n_asian, HQE_asian[0, 5] * 32 * n_asian ** (-1.), 'k:', label='Rate 1')
        if j == 0:
            plt.loglog(n_asian, Weak_asian[0, j, 3] * 8 ** 2 * n_asian ** (-2.), 'k-.', label='Rate 2')
            plt.ylim(0.0004, 0.7)
        if j == 1:
            plt.loglog(n_asian, Weak_asian[0, j, 7] * 128 ** 2 * n_asian ** (-2.), 'k-.', label='Rate 2')
            plt.ylim(0.00007, 0.7)
        plt.xlabel('Number of time steps', fontsize='12')
        plt.title(f'Maximal relative errors of Asian call prices with H={H_asian} and N={N_asian[j]}', fontsize='12')
        plt.legend(loc='best', fontsize='12')
        plt.tight_layout()
        plt.show()
