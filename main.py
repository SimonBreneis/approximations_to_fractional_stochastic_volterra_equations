import time
import numpy as np
import matplotlib.pyplot as plt
import ComputationalFinance as cf
import Data
import rBergomi
import rBergomiMarkov
import rHeston
import rHestonMarkov
import RoughKernel as rk
import rHestonNinomiyaVictoir as nv
import rHestonNV2nd as nv2
import mpmath as mp
import rHestonImplicitEuler as ie
import rHestonSplitKernel as sk
import scipy

nodes, weights = rk.quadrature_rule_geometric_standard(0.1, 2, 1., 'optimized')
dt = np.ones(3000)/3000
times = np.zeros(3001)
times[1:] = np.cumsum(dt)
result = rHestonMarkov.solve_Riccati(2-10j, lambda_=0.3, rho=-0.7, nu=0.3, nodes=nodes, weights=weights, dt=dt)
print(np.interp(0.02, times, result))
dt, _ = rk.adaptive_time_steps(nodes, 1., 10, 200)
times = np.zeros(4001)
times[1:] = np.cumsum(dt)
result = rHestonMarkov.solve_Riccati(2-10j, lambda_=0.3, rho=-0.7, nu=0.3, nodes=nodes, weights=weights, dt=dt)
print(np.interp(0.02, times, result))

dt = np.ones(30000)/30000
times = np.zeros(30001)
times[1:] = np.cumsum(dt)
result = rHestonMarkov.solve_Riccati(2-10j, lambda_=0.3, rho=-0.7, nu=0.3, nodes=nodes, weights=weights, dt=dt)
print(np.interp(0.02, times, result))
dt, _ = rk.adaptive_time_steps(nodes, 1., 10, 2000)
times = np.zeros(40001)
times[1:] = np.cumsum(dt)
result = rHestonMarkov.solve_Riccati(2-10j, lambda_=0.3, rho=-0.7, nu=0.3, nodes=nodes, weights=weights, dt=dt)
print(np.interp(0.02, times, result))


truth = Data.rrHeston_1
recomputed_truth = np.array([0.36679211, 0.37151053, 0.37349597, 0.37300667, 0.37032276, 0.36581155,
 0.36003932, 0.35392561, 0.34881425, 0.34607352, 0.34613224, 0.34811188,
 0.35066277, 0.35275747, 0.35383477, 0.35367447, 0.35228275, 0.34983653,
 0.34666507, 0.34323466, 0.34009206, 0.33772914, 0.33639885, 0.33601041,
 0.33620251, 0.33652615, 0.33659838, 0.33617141, 0.33514212, 0.33353737,
 0.33149203, 0.32922015, 0.32697315, 0.32498276, 0.32339958, 0.32225261,
 0.32145111, 0.32082693, 0.32019298, 0.31939254, 0.3183283 , 0.3169722,
 0.31536127, 0.31358309, 0.31175343, 0.30998849, 0.30837712, 0.30696039,
 0.3057252 , 0.30461362, 0.30354326, 0.30243078, 0.30121143, 0.29985104,
 0.29834961, 0.29673734, 0.29506486, 0.29338986, 0.29176301, 0.29021636,
 0.28875734, 0.2873693 , 0.28601827, 0.28466305, 0.28326577, 0.28180032,
 0.28025685, 0.2786423 , 0.27697698, 0.27528857, 0.27360491, 0.27194739,
 0.27032638, 0.26873978, 0.26717468, 0.2656115 , 0.26402922, 0.26241048,
 0.26074513, 0.25903182, 0.25727743, 0.25549458, 0.25369811, 0.25190129,
 0.25011271, 0.24833462, 0.24656279, 0.2447881 , 0.24299894, 0.24118418,
 0.23933557, 0.23744935, 0.23552668, 0.23357276, 0.23159523, 0.22960199,
 0.2275992 , 0.22558976, 0.22357276, 0.22154379, 0.2194961 , 0.21742222,
 0.21531559, 0.2131719 , 0.2109898 , 0.20877079, 0.20651855, 0.20423768,
 0.20193233, 0.19960503, 0.19725593, 0.19488264, 0.1924807 , 0.19004452,
 0.18756857, 0.18504852, 0.18248203, 0.17986913, 0.17721198, 0.17451422,
 0.17178001, 0.16901299, 0.16621548, 0.16338817, 0.16053031, 0.15764051,
 0.15471793, 0.15176381, 0.14878291, 0.14578461, 0.14278366, 0.13980009,
 0.13685872, 0.13398793, 0.13121828, 0.12858084, 0.12610577, 0.12382093,
 0.12175093, 0.11991626, 0.11833263, 0.11701027, 0.11595315, 0.1151582,
 0.11461458, 0.1143035 , 0.11419869, 0.11426809, 0.11447666, 0.11479004,
 0.11517853, 0.11562054 ,0.11610477, 0.11663058, 0.11720621, 0.11784479,
 0.11855868, 0.11935326 ,0.12022168, 0.12114257, 0.12208199])
untruth = np.array([4.18877678e-01, 4.10908805e-01, 3.95918637e-01, 3.69571621e-01,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 3.33233398e-01, 3.68274123e-01, 3.82462863e-01, 3.88932231e-01,
 3.89972470e-01, 3.86400807e-01, 3.78593612e-01, 3.66766155e-01,
 3.51215599e-01, 3.33343645e-01, 3.20194735e-01, 3.23762569e-01,
 3.36580576e-01, 3.48452289e-01, 3.57060346e-01, 3.62214677e-01,
 3.64073994e-01, 3.62879258e-01, 3.58961358e-01, 3.52822238e-01,
 3.45278897e-01, 3.37655776e-01, 3.31802389e-01, 3.29370881e-01,
 3.30509697e-01, 3.33772062e-01, 3.37410007e-01, 3.40212555e-01,
 3.41545324e-01, 3.41175371e-01, 3.39148558e-01, 3.35738841e-01,
 3.31435426e-01, 3.26922023e-01, 3.22983931e-01, 3.20290407e-01,
 3.19118632e-01, 3.19234909e-01, 3.20053553e-01, 3.20918798e-01,
 3.21304959e-01, 3.20890582e-01, 3.19560744e-01, 3.17385701e-01,
 3.14592313e-01, 3.11523530e-01, 3.08573821e-01, 3.06096405e-01,
 3.04303237e-01, 3.03203600e-01, 3.02617723e-01, 3.02255231e-01,
 3.01811934e-01, 3.01044817e-01, 2.99812725e-01, 2.98088440e-01,
 2.95950569e-01, 2.93559817e-01, 2.91121348e-01, 2.88836608e-01,
 2.86853779e-01, 2.85231641e-01, 2.83930194e-01, 2.82830780e-01,
 2.81775665e-01, 2.80611392e-01, 2.79223484e-01, 2.77556833e-01,
 2.75621163e-01, 2.73483139e-01, 2.71247538e-01, 2.69030991e-01,
 2.66933630e-01, 2.65015499e-01, 2.63284206e-01, 2.61697034e-01,
 2.60175868e-01, 2.58629385e-01, 2.56975784e-01, 2.55160609e-01,
 2.53166547e-01, 2.51014235e-01, 2.48754717e-01, 2.46455558e-01,
 2.44183796e-01, 2.41989733e-01, 2.39895519e-01, 2.37891283e-01,
 2.35939419e-01, 2.33985292e-01, 2.31971162e-01, 2.29849674e-01,
 2.27593901e-01, 2.25202027e-01, 2.22696027e-01, 2.20114932e-01,
 2.17504368e-01, 2.14904854e-01, 2.12341666e-01, 2.09818669e-01,
 2.07317495e-01, 2.04802054e-01, 2.02227014e-01, 1.99548081e-01,
 1.96731599e-01, 1.93761349e-01, 1.90641082e-01, 1.87392272e-01,
 1.84047599e-01, 1.80641616e-01, 1.77200737e-01, 1.73734912e-01,
 1.70232994e-01, 1.66663022e-01, 1.62977558e-01, 1.59123162e-01,
 1.55052272e-01, 1.50735260e-01, 1.46170384e-01, 1.41389649e-01,
 1.36459316e-01, 1.31474882e-01, 1.26551514e-01, 1.21812003e-01,
 1.17374774e-01, 1.13344287e-01, 1.09805144e-01, 1.06819902e-01,
 1.04429341e-01, 1.02653257e-01, 1.01490012e-01, 1.00914081e-01,
 1.00872500e-01, 1.01282883e-01, 1.02036310e-01, 1.03007108e-01,
 1.04068559e-01, 1.05110847e-01, 1.06056778e-01, 1.06872098e-01,
 1.07569167e-01, 1.08203892e-01, 1.08865547e-01, 1.09658260e-01,
 1.10673678e-01, 1.11958914e-01, 1.13491305e-01, 1.15173999e-01,
 1.16857138e-01, 1.18374790e-01, 1.19581958e-01, 1.20382597e-01,
 1.20749756e-01])

rerecomputed_truth = np.array([0.36679211, 0.37151053, 0.37349597, 0.37300667, 0.37032276, 0.36581155,
 0.36003932, 0.35392561, 0.34881425, 0.34607352, 0.34613224, 0.34811188,
 0.35066277, 0.35275747, 0.35383477, 0.35367447, 0.35228275, 0.34983653,
 0.34666507, 0.34323466, 0.34009206, 0.33772914, 0.33639885, 0.33601041,
 0.33620251, 0.33652615, 0.33659838, 0.33617141, 0.33514212, 0.33353737,
 0.33149203, 0.32922015, 0.32697315, 0.32498276, 0.32339958, 0.32225261,
 0.32145111, 0.32082693, 0.32019298, 0.31939254, 0.3183283 , 0.3169722,
 0.31536127 ,0.31358309, 0.31175343, 0.30998849, 0.30837712, 0.30696039,
 0.3057252  ,0.30461362, 0.30354326, 0.30243078, 0.30121143, 0.29985104,
 0.29834961 ,0.29673734, 0.29506486, 0.29338986, 0.29176301, 0.29021636,
 0.28875734 ,0.2873693 , 0.28601827, 0.28466305, 0.28326577, 0.28180032,
 0.28025685 ,0.2786423 , 0.27697698, 0.27528857, 0.27360491, 0.27194739,
 0.27032638, 0.26873978, 0.26717468, 0.2656115 , 0.26402922, 0.26241048,
 0.26074513, 0.25903182, 0.25727743, 0.25549458, 0.25369811, 0.25190129,
 0.25011271, 0.24833462, 0.24656279, 0.2447881 , 0.24299894, 0.24118418,
 0.23933557, 0.23744935, 0.23552668, 0.23357276, 0.23159523, 0.22960199,
 0.2275992 , 0.22558976, 0.22357276, 0.22154379, 0.2194961 , 0.21742222,
 0.21531559, 0.2131719 , 0.2109898 , 0.20877079, 0.20651855, 0.20423768,
 0.20193233, 0.19960503, 0.19725593, 0.19488264, 0.1924807 , 0.19004452,
 0.18756857, 0.18504852, 0.18248203, 0.17986913, 0.17721198, 0.17451422,
 0.17178001, 0.16901299, 0.16621548, 0.16338817, 0.16053031, 0.15764051,
 0.15471793, 0.15176381, 0.14878291, 0.14578461, 0.14278366, 0.13980009,
 0.13685872, 0.13398793, 0.13121828, 0.12858084 ,0.12610577, 0.12382093,
 0.12175093, 0.11991626, 0.11833263, 0.11701027, 0.11595315, 0.1151582,
 0.11461458, 0.1143035 , 0.11419869, 0.11426809, 0.11447666, 0.11479004,
 0.11517853, 0.11562054, 0.11610477, 0.11663058, 0.11720621, 0.11784479,
 0.11855868, 0.11935326, 0.12022168, 0.12114257, 0.12208199])

totally_true = Data.rrHeston
optimized_points = np.array([0.40286746, 0.40315553, 0.40308528, 0.40241364, 0.40105379, 0.39906079,
 0.39660925, 0.39396115, 0.39141543, 0.38923719, 0.3875829 , 0.38645551,
 0.3857155 , 0.38514044, 0.38449857, 0.38360628, 0.38235887, 0.38073907,
 0.37880991, 0.37669597, 0.37455437, 0.37253804, 0.37075797, 0.3692561,
 0.36799891, 0.36689387, 0.36582057, 0.36466407, 0.36334071, 0.36181249,
 0.36008982, 0.35822465, 0.3562958 , 0.35438929, 0.35257758, 0.35090261,
 0.3493676 , 0.34793959, 0.34656135, 0.34516803, 0.34370336, 0.34213174,
 0.34044398, 0.33865677, 0.33680658, 0.33493966, 0.33310043, 0.33132103,
 0.32961446, 0.32797299, 0.32637195, 0.32477729, 0.32315465, 0.32147744,
 0.31973222, 0.31792053, 0.31605715, 0.3141655 , 0.3122714 , 0.31039673,
 0.30855452, 0.30674647, 0.30496351, 0.30318891, 0.30140294, 0.29958774,
 0.2977314 , 0.29583009, 0.29388826, 0.29191685, 0.2899301 , 0.28794191,
 0.28596245, 0.28399607, 0.28204059, 0.28008839, 0.27812858, 0.27614985,
 0.27414313, 0.27210347, 0.27003084, 0.26792966, 0.26580735, 0.2636723,
 0.26153167, 0.25938975, 0.25724707, 0.25510048, 0.25294413, 0.25077098,
 0.2485745 , 0.2463501 , 0.24409604, 0.2418135 , 0.23950607, 0.23717857,
 0.23483579, 0.23248119, 0.2301161 , 0.22773945, 0.22534805, 0.22293743,
 0.22050288, 0.21804053, 0.21554807, 0.21302519, 0.21047341, 0.20789551,
 0.2052947 , 0.20267375, 0.20003421, 0.19737604, 0.19469759, 0.19199604,
 0.18926804, 0.18651058, 0.18372166, 0.18090081, 0.17804921, 0.17516946,
 0.17226507, 0.16933982, 0.16639709, 0.16343944, 0.16046851, 0.15748529,
 0.15449079, 0.15148688, 0.14847735, 0.14546874, 0.14247103, 0.13949786,
 0.13656639, 0.13369676, 0.13091121, 0.1282331 , 0.12568588, 0.12329213,
 0.12107287, 0.11904694, 0.11723061, 0.11563707, 0.11427605, 0.11315308,
 0.11226869, 0.11161754, 0.11118766, 0.11096035, 0.11091089, 0.11101049,
 0.11122925, 0.11153981, 0.11192098, 0.11236044, 0.11285587, 0.11341391,
 0.11404667, 0.11476618, 0.11557767, 0.11647376, 0.11743149])

optimized_adaptive = np.array([0.40796192, 0.40570888, 0.39998258, 0.39115147, 0.3798494 , 0.36761238,
 0.35809632, 0.35616493, 0.36146764, 0.36905497, 0.37563322, 0.37999488,
 0.38181471, 0.38110126, 0.37804196, 0.37300035, 0.3665917 , 0.35980204,
 0.35401945, 0.35064748, 0.35022132, 0.35197847, 0.35454007, 0.35673227,
 0.35783957 ,0.35753156, 0.35575523, 0.35267488, 0.34865217, 0.34423441,
 0.34010045 ,0.33690602, 0.33503879, 0.33444014, 0.3346629 , 0.33510944,
 0.33525094 ,0.33473124, 0.3333849 , 0.33122182, 0.32840412, 0.32521661,
 0.32201915 ,0.31916986, 0.31692669, 0.31536623, 0.31436784, 0.31367363,
 0.31298441 ,0.31204294, 0.31068248, 0.30884415, 0.30657365, 0.30400371,
 0.30132379 ,0.29873739, 0.29641243, 0.29443779, 0.29280291, 0.291409,
 0.29010556 ,0.28873538, 0.28717253, 0.28534602, 0.28324818, 0.2809301,
 0.27848624 ,0.27603087, 0.27367032, 0.27147721, 0.269474  , 0.26763092,
 0.26587824, 0.26412781, 0.26229619, 0.26032359, 0.25818479, 0.25589151,
 0.25348664, 0.25103218, 0.24859343, 0.24622312, 0.24394922, 0.24176972,
 0.23965542, 0.23755937, 0.23542971, 0.23322238, 0.23091064, 0.22848965,
 0.22597579, 0.2234012 , 0.22080499, 0.21822338, 0.21568099, 0.21318551,
 0.21072681, 0.20828062, 0.20581528, 0.20329977, 0.20071093, 0.19803818,
 0.19528505, 0.19246729, 0.18960816, 0.18673226, 0.1838593 , 0.18099934,
 0.17815069, 0.17530075, 0.17242944, 0.16951434, 0.16653604, 0.16348251,
 0.16035158, 0.15715091, 0.15389566, 0.15060448, 0.14729477, 0.14397838,
 0.14065892, 0.13733119, 0.13398303, 0.13059902, 0.12716541, 0.12367504,
 0.12013136, 0.11655063, 0.11296193, 0.10940505, 0.10592674, 0.1025762,
 0.09940094, 0.09644363, 0.09374068, 0.09132229, 0.08921389, 0.0874381,
 0.08601636, 0.08496928, 0.08431434, 0.08406034, 0.08419861, 0.08469374,
 0.08547838, 0.08645708, 0.08752037, 0.08856512, 0.08951509, 0.09033709,
 0.09105132, 0.09173548, 0.09251861, 0.09355442, 0.0949652 , 0.09677184,
 0.09885898, 0.10100826, 0.10297012, 0.1045213 , 0.10548845])

optimized_adaptive_reversed = np.array([0.41765379, 0.4167913,  0.41599367 ,0.41510882, 0.4140228 , 0.41267901,
 0.41108272, 0.40929368, 0.40740934, 0.40554104, 0.40378721, 0.40220974,
 0.4008206 , 0.3995829 , 0.39842525, 0.39726347, 0.39602226, 0.39465147,
 0.39313442, 0.3914881 , 0.38975636, 0.38799787, 0.38627143, 0.38462194,
 0.38307051, 0.38161113, 0.38021464, 0.37883815, 0.37743671, 0.37597416,
 0.37443055, 0.37280504, 0.37111434, 0.36938731, 0.36765725, 0.36595366,
 0.3642955 , 0.36268746, 0.36112004, 0.35957313, 0.35802178, 0.35644259,
 0.35481902, 0.35314461, 0.35142348, 0.34966848, 0.34789737, 0.34612824,
 0.34437515, 0.34264508, 0.34093684, 0.33924202, 0.3375476 , 0.33583946,
 0.33410575, 0.33233948, 0.33053967, 0.32871107, 0.3268625 , 0.32500442,
 0.32314625, 0.32129422, 0.31945008, 0.317611  , 0.31577064, 0.31392084,
 0.31205374, 0.31016352, 0.30824761, 0.30630696, 0.30434548, 0.30236887,
 0.30038304, 0.29839265, 0.29640007, 0.2944049 , 0.29240426, 0.2903936,
 0.28836788, 0.28632272, 0.28425536, 0.28216513, 0.2800534 , 0.27792301,
 0.27577747, 0.27361998, 0.27145261, 0.26927586, 0.26708854, 0.26488813,
 0.26267139, 0.26043512, 0.25817685, 0.25589531, 0.25359059, 0.2512639,
 0.24891721, 0.24655259, 0.24417164, 0.24177507, 0.23936247, 0.23693242,
 0.23448278, 0.23201121, 0.22951562, 0.22699462, 0.22444772, 0.22187536,
 0.21927867, 0.2166591 , 0.214018  , 0.21135626, 0.20867405, 0.20597079,
 0.20324529, 0.20049612, 0.19772195, 0.19492196, 0.19209613, 0.18924537,
 0.18637146, 0.18347684, 0.18056433, 0.17763683, 0.17469709, 0.17174769,
 0.16879114, 0.16583027, 0.16286869, 0.15991136, 0.15696511, 0.15403905,
 0.15114476, 0.14829626, 0.14550973, 0.14280295, 0.14019473, 0.13770416,
 0.13534991, 0.13314957, 0.13111912, 0.12927237, 0.12762056, 0.1261719,
 0.12493113, 0.1238991 , 0.12307233, 0.12244283, 0.12199816, 0.12172204,
 0.12159534, 0.1215978 , 0.12170982, 0.12191448, 0.12219902, 0.12255562,
 0.12298116, 0.12347583, 0.12404082, 0.12467548, 0.12537466])

optimized_5000_steps = np.array([0.40286717, 0.4031553 , 0.40308515, 0.40241363, 0.4010539 , 0.39906098,
 0.39660948, 0.39396136, 0.39141558, 0.38923724, 0.38758286, 0.38645539,
 0.38571535, 0.38514029, 0.38449847, 0.38360624, 0.38235889, 0.38073915,
 0.37881003, 0.37669609, 0.37455447, 0.37253809, 0.37075797, 0.36925606,
 0.36799883, 0.36689379, 0.3658205 , 0.36466402, 0.3633407 , 0.36181251,
 0.36008987, 0.35822471, 0.35629585, 0.35438933, 0.35257759, 0.3509026,
 0.34936756, 0.34793954, 0.34656131, 0.34516799, 0.34370334, 0.34213174,
 0.340444  , 0.3386568 , 0.33680661, 0.33493968, 0.33310044, 0.33132102,
 0.32961444, 0.32797296, 0.32637192, 0.32477726, 0.32315463, 0.32147743,
 0.31973222, 0.31792053, 0.31605716, 0.31416551, 0.3122714 , 0.31039673,
 0.30855451, 0.30674645, 0.30496349, 0.30318889, 0.30140292, 0.29958773,
 0.29773139, 0.29583008, 0.29388826, 0.29191685, 0.2899301 , 0.2879419,
 0.28596245, 0.28399606, 0.28204057, 0.28008837, 0.27812856, 0.27614984,
 0.27414313, 0.27210347, 0.27003084, 0.26792966, 0.26580736, 0.2636723,
 0.26153167, 0.25938974, 0.25724706, 0.25510047, 0.25294413, 0.25077097,
 0.24857449, 0.2463501 , 0.24409604, 0.2418135 , 0.23950607, 0.23717858,
 0.23483579, 0.23248119, 0.2301161 , 0.22773945, 0.22534805, 0.22293743,
 0.22050289, 0.21804053, 0.21554807, 0.2130252 , 0.21047342, 0.20789552,
 0.20529471, 0.20267376, 0.20003422, 0.19737605, 0.19469761, 0.19199605,
 0.18926806, 0.1865106 , 0.18372168, 0.18090084, 0.17804924, 0.17516948,
 0.1722651 , 0.16933985, 0.16639712, 0.16343948, 0.16046855, 0.15748533,
 0.15449082, 0.15148692, 0.14847739, 0.14546879, 0.14247108, 0.1394979,
 0.13656643, 0.13369679, 0.13091124, 0.12823312, 0.12568588, 0.12329212,
 0.12107284, 0.11904689, 0.11723053, 0.11563698, 0.11427594, 0.11315296,
 0.11226856, 0.1116174 , 0.11118752, 0.11096021, 0.11091076, 0.11101037,
 0.11122914, 0.11153972, 0.1119209 , 0.11236037, 0.1128558 , 0.11341384,
 0.11404659, 0.1147661 , 0.11557759, 0.11647368, 0.11743143])

optimized_adaptive_500_steps = np.array([0.40796342, 0.40571018, 0.39998345, 0.39115163, 0.37984851, 0.36761007,
 0.35809276, 0.35616164, 0.36146585, 0.36905451, 0.37563358, 0.37999568,
 0.38181566, 0.38110218, 0.37804267, 0.37300069, 0.36659154, 0.35980134,
 0.35401831, 0.35064622, 0.35022032, 0.35197795, 0.35454002, 0.35673257,
 0.35784007, 0.35753211, 0.35575571, 0.35267519, 0.34865222, 0.34423418,
 0.34009998, 0.33690541, 0.33503822, 0.33443974, 0.33466272, 0.33510948,
 0.33525114, 0.33473152, 0.33338518, 0.33122203, 0.32840421, 0.32521654,
 0.32201894, 0.31916955, 0.31692636, 0.31536594, 0.31436766, 0.31367356,
 0.31298445, 0.31204306, 0.31068263, 0.30884428, 0.30657372, 0.3040037,
 0.30132369, 0.29873723, 0.29641223, 0.2944376 , 0.29280275, 0.2914089,
 0.29010553, 0.2887354 , 0.28717259, 0.28534608, 0.28324822, 0.2809301,
 0.2784862 , 0.27603078, 0.2736702 , 0.27147709, 0.26947389, 0.26763083,
 0.26587819, 0.26412779, 0.26229621, 0.26032361, 0.25818481, 0.25589151,
 0.25348662, 0.25103213, 0.24859336, 0.24622304, 0.24394914, 0.24176965,
 0.23965538, 0.23755935, 0.23542971, 0.23322239, 0.23091065, 0.22848965,
 0.22597579, 0.22340118, 0.22080495, 0.21822333, 0.21568095, 0.21318547,
 0.21072678, 0.2082806 , 0.20581528, 0.20329979, 0.20071095, 0.1980382,
 0.19528507, 0.1924673 , 0.18960816, 0.18673226, 0.18385929, 0.18099933,
 0.1781507 , 0.17530076, 0.17242947, 0.16951438, 0.16653609, 0.16348257,
 0.16035164, 0.15715096 ,0.15389571, 0.15060454, 0.14729483, 0.14397845,
 0.140659  , 0.13733128, 0.13398313, 0.13059914, 0.12716554, 0.12367517,
 0.12013149, 0.11655074, 0.11296201, 0.10940511, 0.10592675, 0.10257617,
 0.09940087, 0.09644352, 0.09374053, 0.09132211, 0.0892137 , 0.0874379,
 0.08601617, 0.0849691 , 0.08431419, 0.08406023, 0.08419855, 0.08469373,
 0.08547841, 0.08645715, 0.08752045, 0.08856519, 0.08951513, 0.09033709,
 0.09105128, 0.0917354 , 0.09251851, 0.09355434, 0.09496517, 0.0967719,
 0.09885912, 0.10100846, 0.10297034, 0.10452149, 0.10548856])

new_optimized_adaptive = np.array([0.40286706, 0.40315486, 0.40308454, 0.402413  , 0.40105339, 0.39906068,
 0.39660945, 0.39396159, 0.391416  , 0.38923774, 0.3875833 , 0.38645567,
 0.38571542, 0.38514016, 0.3844982 , 0.3836059 , 0.38235858, 0.38073893,
 0.37880994, 0.37669615, 0.37455466, 0.37253835, 0.37075824, 0.36925627,
 0.36799895, 0.36689379, 0.36582041, 0.36466387, 0.36334054, 0.36181238,
 0.3600898 , 0.35822472, 0.35629594, 0.35438947, 0.35257775, 0.35090275,
 0.34936767, 0.34793959, 0.3465613 , 0.34516794, 0.34370327, 0.34213167,
 0.34044396, 0.3386568 , 0.33680665, 0.33493976, 0.33310054, 0.33132113,
 0.32961453, 0.32797302, 0.32637195, 0.32477727, 0.32315462, 0.32147741,
 0.3197322 , 0.31792054, 0.31605719, 0.31416556, 0.31227147, 0.31039681,
 0.30855458, 0.30674651, 0.30496353, 0.30318892, 0.30140293, 0.29958774,
 0.29773139, 0.29583009, 0.29388828, 0.29191689, 0.28993015, 0.28794196,
 0.2859625 , 0.28399611, 0.28204062, 0.2800884 , 0.27812858, 0.27614986,
 0.27414314, 0.27210348, 0.27003085, 0.26792968, 0.26580739, 0.26367234,
 0.2615317 , 0.25938978, 0.25724709, 0.2551005 , 0.25294414, 0.25077098,
 0.24857449, 0.2463501 , 0.24409604, 0.24181351, 0.23950608, 0.23717859,
 0.2348358 , 0.2324812 , 0.23011611, 0.22773945, 0.22534804, 0.22293742,
 0.22050287, 0.21804051, 0.21554805, 0.21302518, 0.2104734 , 0.20789549,
 0.20529469, 0.20267373, 0.20003419, 0.19737601, 0.19469756, 0.19199599,
 0.18926799, 0.18651053, 0.1837216 , 0.18090075, 0.17804915, 0.17516939,
 0.17226501, 0.16933975, 0.16639702, 0.16343937, 0.16046843, 0.15748521,
 0.1544907 , 0.1514868 , 0.14847728, 0.14546869, 0.142471  , 0.13949785,
 0.13656641, 0.1336968 , 0.13091129, 0.12823321, 0.125686  , 0.12329226,
 0.121073  , 0.11904707, 0.11723072, 0.11563716, 0.11427611, 0.1131531,
 0.11226867, 0.11161747, 0.11118756, 0.11096022, 0.11091075, 0.11101034,
 0.1112291 , 0.11153968, 0.11192086, 0.11236034, 0.11285578, 0.11341383,
 0.11404658, 0.11476607, 0.11557754, 0.1164736 , 0.11743133])

print('is it better? ', np.sum(np.abs(new_optimized_adaptive - optimized_adaptive)))

plt.plot(Data.k_rrHeston, totally_true, label='True rough Heston')
plt.plot(Data.k_rrHeston, Data.rrHeston_1, label='observation, 2 points')
plt.plot(Data.k_rrHeston, optimized_points, label='optimized, 2 points')
plt.plot(Data.k_rrHeston, optimized_adaptive, label='optimized adaptive, 2 points')
plt.plot(Data.k_rrHeston, optimized_adaptive_reversed, label='optimized adaptive reversed, 2 points')
plt.plot(Data.k_rrHeston, optimized_5000_steps, label='optimized 5000 steps, 2 points')
plt.plot(Data.k_rrHeston, optimized_adaptive_500_steps, label='optimized adaptive 500 steps, 2 points')
plt.plot(Data.k_rrHeston, new_optimized_adaptive, 'k-', label='new optimized adaptive, 2 points')
plt.legend(loc='upper right')
plt.show()

print(np.sum(np.abs(recomputed_truth - truth)))
print(np.sum(np.abs(untruth - truth)))
print(np.sum(np.abs(rerecomputed_truth - truth)))
# print(rHestonMarkov.implied_volatility(K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, adaptive=True, mode='optimized'))
print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=200, adaptive=True, R=2, N_fourier=10000, L=50))
# print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
# print(rHestonMarkov.implied_volatility(N=1, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
time.sleep(360000)

print(rk.quadrature_rule_geometric_good(0.1, 1, 1., 'observation'))
print(rk.quadrature_rule_geometric_standard(0.1, 1, 1., 'observation'))
Data.rHeston_smiles_precise()
print(rHestonMarkov.implied_volatility(N=1, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
time.sleep(360000)
print(rHeston.implied_volatility(K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02))
time.sleep(360000)
print(rHestonMarkov.implied_volatility(N=1, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1.,
                                       adaptive=False, N_Riccati=1000, R=2., L=200., N_fourier=40000, mode="observation"))
time.sleep(36000)

rk.compare_approximations(0.1, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]), T=1.)
time.sleep(3600000)

print(rHestonMarkov.implied_volatility(Data.k_rrHeston, 0.1, 0.3, -0.7, 0.3, 0.02, 0.02, 1., 1, R=-2, N_Riccati=1000, L=10., N_fourier=1000))
time.sleep(360000)

'''
def smooth_point_distribution(x, eps):
    dim = len(x[0])
    return lambda y: np.sum(np.array([np.exp(-1/2 * np.sum((x[i]-y)**2)/eps**2)/np.sqrt((2*np.pi*eps**2)**dim) for i in range(len(x))]))


x = [np.array([0, 0]), np.array([1, 2])]
x_left, x_right = -1, 3
y_left, y_right = -3, 3
dx = 0.05
distribution = smooth_point_distribution(x, dx)

# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(np.arange(y_left, y_right, dx), np.arange(x_left, x_right, dx))

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = np.array([[distribution(np.array([x[i, j], y[i, j]])) for j in range(x.shape[1])] for i in range(x.shape[0])])
z = z[:-1, :-1]
z_min, z_max = z.min(), z.max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='Reds', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()

with open('dW0.npy', 'rb') as f:
    dW = np.load(f)
with open('dB0.npy', 'rb') as f:
    dB = np.load(f)

WB = np.empty((2, 1, 2048))
WB[0, 0, :] = dW[0, :]
WB[1, 0, :] = dB[0, :]
S_1, V_1, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, WB=WB, vol_behaviour='adaptive')
plt.plot(np.linspace(0, 1, 2049), S_1[0, :], label='stock price, N=2048')
plt.plot(np.linspace(0, 1, 2049), V_1[0, :], label='volatility, N=2048')
plt.plot(np.linspace(0, 1, 1025), S_2[0, :], label='stock price, N=1024')
plt.plot(np.linspace(0, 1, 1025), V_2[0, :], label='volatility, N=1024')
plt.legend(loc='best')
plt.xlabel('t')
plt.title(f'Sample paths of {vb} implementation')
plt.show()



time.sleep(3600)
'''
''''
S, V, _, bad = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1, N=6, vol_behaviour='adaptive')
print(bad)
plt.plot(np.linspace(0, 1, 1001), S[0, :])
plt.plot(np.linspace(0, 1, 1001), V[0, :])
plt.show()
ie.call(K=np.array([1.]), vol_behaviour='adaptive')
# methods = ['mean reversion', 'hyperplane reflection', 'split throw', 'multiple time scales']
# methods = ['sticky', 'hyperplane reset', 'split kernel']
'''
methods = ['adaptive']

with open('dW0.npy', 'rb') as f:
    dW = np.load(f)
with open('dB0.npy', 'rb') as f:
    dB = np.load(f)
'''
WB_1 = np.empty((2, 1, 2048))
WB_2 = np.empty((2, 1, 1024))
WB_1[0, 0, :] = dW[0, :]
WB_1[1, 0, :] = dB[0, :]
for i in range(1024):
    WB_2[:, :, i] = WB_1[:, :, 2*i] + WB_1[:, :, 2*i+1]
for vb in methods:
    S_1, V_1, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, WB=WB_1, vol_behaviour=vb)
    S_2, V_2, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, WB=WB_2, vol_behaviour=vb)
    plt.plot(np.linspace(0, 1, 2049), S_1[0, :], label='stock price, N=2048')
    plt.plot(np.linspace(0, 1, 2049), V_1[0, :], label='volatility, N=2048')
    plt.plot(np.linspace(0, 1, 1025), S_2[0, :], label='stock price, N=1024')
    plt.plot(np.linspace(0, 1, 1025), V_2[0, :], label='volatility, N=1024')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.title(f'Sample paths of {vb} implementation')
    plt.show()
'''
'''
for vb in methods:
    samples = 10000
    S_errors = np.empty(samples)
    V_errors = np.empty(samples)
    WB_1 = np.empty((2, samples, 2048))
    WB_2 = np.empty((2, samples, 1024))
    WB_1[0, :, :] = dW[:samples, :]
    WB_1[1, :, :] = dB[:samples, :]
    for j in range(1024):
        WB_2[:, :, j] = WB_1[:, :, 2 * j] + WB_1[:, :, 2 * j + 1]
    S_1_, V_1_, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6,
                                     WB=WB_1, vol_behaviour=vb)
    S_2, V_2, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6,
                                     WB=WB_2, vol_behaviour=vb)
    S_1 = np.empty((samples, 1025))
    V_1 = np.empty((samples, 1025))
    for i in range(1025):
        S_1[:, i] = S_1_[:, 2*i]
        V_1[:, i] = V_1_[:, 2*i]
    S_errors = np.amax(np.abs(S_1-S_2), axis=-1)
    V_errors = np.amax(np.abs(V_1-V_2), axis=-1)
    print(len(S_errors))
    S_avg_error, S_std_error = cf.MC(S_errors)
    V_avg_error, V_std_error = cf.MC(V_errors)
    print(f'The strong error for S is roughly {S_avg_error} +/- {S_std_error}.')
    print(f'The strong error for V is roughly {V_avg_error} +/- {V_std_error}.')
'''
N_time = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
samples = np.empty(1000000)
k_vec = Data.k_rrHeston
S = np.empty((len(N_time), 1000000))

print("Here")

for vb in methods:
    for N_t in N_time:
        WB = np.empty((2, 100000, N_t))
        factor = 2048/N_t
        for i in range(10):
            with open(f'dW{i}.npy', 'rb') as f:
                dW = np.load(f)
            with open(f'dB{i}.npy', 'rb') as f:
                dB = np.load(f)
            for j in range(N_t):
                WB[0, :, j] = np.sum(dW[:, int(j*factor):int((j+1)*factor)], axis=-1)
                WB[1, :, j] = np.sum(dB[:, int(j*factor):int((j+1)*factor)], axis=-1)
            print(vb, N_t, i)
            samples[i*100000:(i+1)*100000] = ie.samples(H=0.49, N=1, N_time=N_t, WB=WB, vol_behaviour=vb)
        np.save(f'samples of {vb} mode with N_time={N_t} and H=0.49 and N=1', samples)
time.sleep(3600000)
for vb in methods:
    for i in range(len(N_time)):
        print(N_time[i])
        with open(f'samples of {vb} mode with N_time={N_time[i]}.npy', 'rb') as f:
            S[i, :] = np.load(f)
        est, lower, upper = cf.volatility_smile_call(S[i, :], np.exp(k_vec), 1., 1.)
        plt.plot(k_vec, est, label=f'N_time={N_time[i]}')
    plt.plot(k_vec, lower, 'k--')
    plt.plot(k_vec, upper, 'k--')
    plt.plot(Data.k_rrHeston, Data.rrHeston_6, 'k-', label=f'Fourier inversion')
    plt.title(vb)
    plt.legend(loc='best')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.show()

strikes = np.array([-0.3, 0., 0.15])
true_iv = np.array([np.interp(strike, Data.k_rrHeston, Data.rrHeston_6) for strike in strikes])
all_errors = np.empty((len(methods), len(N_time), len(strikes)))
for k in range(len(methods)):
    conv_rate = 1.
    for i in range(len(N_time)):
        print(N_time[i])
        with open(f'samples of {methods[k]} mode with N_time={N_time[i]}.npy', 'rb') as f:
            S[i, :] = np.load(f)
    S_Rich = (S[1:, :] - 2 ** (-conv_rate) * S[:-1, :]) / (1 - 2 ** (-conv_rate))
    iv = np.empty((3, len(N_time), len(strikes)))
    iv_Rich = np.empty((3, len(N_time)-1, len(strikes)))
    errors = np.empty((len(N_time), len(strikes)))
    errors_Rich = np.empty((len(N_time)-1, len(strikes)))
    errors_alt = np.empty((len(N_time)-1, len(strikes)))
    confidence = np.empty(len(strikes))
    confidence_Rich = np.empty(len(strikes))
    for i in range(len(N_time)):
        print(N_time[i])
        iv[0, i, :], iv[1, i, :], iv[2, i, :] = cf.volatility_smile_call(S[i, :], np.exp(strikes), 1., 1.)
        errors[i, :] = np.abs(iv[0, i, :] - true_iv)
        all_errors[k, :, :] = errors
    for i in range(len(N_time)-1):
        errors_alt[i, :] = np.abs(iv[0, i, :] - iv[0, -1, :])
        print(N_time[i+1])
        iv_Rich[0, i, :], iv_Rich[1, i, :], iv_Rich[2, i, :] = cf.volatility_smile_call(S_Rich[i, :], np.exp(strikes), 1., 1.)
        # iv_Rich[0, :, :] = (iv[0, 1:, :] - 2**conv_rate * iv[0, :-1, :])/(1-2**conv_rate)
        errors_Rich[i, :] = np.abs(iv_Rich[0, i, :] - true_iv)
    confidence = np.fmax(iv[2, -1, :]-iv[0, -1, :], iv[0, -1, :] - iv[1, -1, :])
    confidence_Rich = np.fmax(iv_Rich[2, -1, :] - iv_Rich[0, -1, :], iv_Rich[0, -1, :] - iv_Rich[1, -1, :])
    constants = np.empty(len(strikes))
    rates = np.empty(len(strikes))
    constants_alt = np.empty(len(strikes))
    rates_alt = np.empty(len(strikes))
    constants_Rich = np.empty(len(strikes))
    rates_Rich = np.empty(len(strikes))
    for i in range(len(strikes)):
        rates[i], constants[i], _, _, _ = Data.log_linear_regression(N_time[3:-1], errors[3:-1, i])
        rates_alt[i], constants_alt[i], _, _, _ = Data.log_linear_regression(N_time[3:-2], errors_alt[3:-1, i])
        rates_Rich[i], constants_Rich[i], _, _, _ = Data.log_linear_regression(N_time[4:], errors_Rich[3:, i])
    for i in range(len(strikes)):
        plt.loglog(N_time, errors[:, i], 'b-', label='error')
        plt.loglog(N_time, constants[i]*N_time**rates[i], 'b--')
        plt.loglog(N_time[1:], errors_Rich[:, i], 'r-', label='error Richardson')
        plt.loglog(N_time, constants_Rich[i]*N_time**rates_Rich[i], 'r--')
        plt.loglog(N_time, confidence[i]*np.ones(len(N_time)), 'k--', label='Monte Carlo error')
        plt.legend(loc='best')
        plt.title(f'{methods[k]}, log-moneyness={strikes[i]},\ncomparison with Fourier inversion')
        x_label = 'number of time steps'
        x_label += '\n\nerror ' + r'$\approx$' + f' {constants[i]:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(rates[i]))
        plt.xlabel(x_label)
        plt.ylabel('error')
        plt.show()

        plt.loglog(N_time, errors[:, i], 'b-', label='total error')
        plt.loglog(N_time, constants[i] * N_time ** rates[i], 'b--')
        plt.loglog(N_time, confidence[i] * np.ones(len(N_time)), 'k--', label='Monte Carlo error')
        plt.legend(loc='best')
        plt.title(f'{methods[k]}, log-moneyness={strikes[i]},\ncomparison with Fourier inversion')
        x_label = 'number of time steps'
        x_label += '\n\nerror ' + r'$\approx$' + f' {constants[i]:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(rates[i]))
        plt.xlabel(x_label)
        plt.ylabel('error')
        plt.show()

        plt.loglog(N_time[:-1], errors_alt[:, i], 'b-', label='error')
        plt.loglog(N_time[:-1], constants_alt[i] * N_time[:-1] ** rates_alt[i], 'b--', label='regression')
        plt.loglog(N_time[:-1], confidence[i]*np.ones(len(N_time[:-1])), 'k--', label='Monte Carlo error')
        plt.legend(loc='best')
        plt.title(f'{methods[k]}, log-moneyness={strikes[i]},\ncomparison with {N_time[-1]} time points')
        x_label = 'number of time steps'
        x_label += '\n\nerror ' + r'$\approx$' + f' {constants_alt[i]:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(rates_alt[i]))
        plt.xlabel(x_label)
        plt.ylabel('error')
        plt.show()

for i in range(len(strikes)):
    for k in range(len(methods)):
        plt.loglog(N_time, all_errors[k, :, i], label=methods[k])
    plt.loglog(N_time, confidence[i] * np.ones(len(N_time)), 'k--', label='Monte Carlo error')
    plt.legend(loc='best')
    plt.title(f'log-moneyness={strikes[i]},\ncomparison with Fourier inversion')
    plt.xlabel('number of time steps')
    plt.ylabel('error')
    plt.show()


time.sleep(3600000)

for vb in methods:
    for N_t in N_time:
        WB = np.empty((2, 100000, N_t))
        factor = 2048/N_t
        for i in range(10):
            with open(f'dW{i}.npy', 'rb') as f:
                dW = np.load(f)
            with open(f'dB{i}.npy', 'rb') as f:
                dB = np.load(f)
            for j in range(N_t):
                WB[0, :, j] = np.sum(dW[:, int(j*factor):int((j+1)*factor)], axis=-1)
                WB[1, :, j] = np.sum(dB[:, int(j*factor):int((j+1)*factor)], axis=-1)
            print(vb, N_t, i)
            samples[i*100000:(i+1)*100000] = ie.samples(N_time=N_t, WB=WB, vol_behaviour=vb)
        np.save(f'samples of {vb} mode with N_time={N_t}', samples)

time.sleep(3600000)

Data.plot_rHeston_IE_smiles()

K = np.exp(Data.k_rrHeston)
N = 6
tic = time.perf_counter()
vol, lower, upper = ie.call(K, N=N, N_time=1000, m=200000, vol_behaviour='split kernel')
toc = time.perf_counter()
print(toc-tic)
print(vol)
print(lower)
print(upper)
time.sleep(360000)

S, V, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, vol_behaviour='split kernel')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of split kernel implementation')
plt.show()

time.sleep(360000)

'''
S, V, _ = ie.get_sample_path(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, vol_behaviour='multiple time scales')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of multiple time scales implementation')
plt.show()
'''

K = np.exp(Data.k_rrHeston)
N = 6
tic = time.perf_counter()
vol, lower, upper = ie.call(K, N=N, N_time=1000, m=200000, vol_behaviour='multiple time scales')
toc = time.perf_counter()
print(toc-tic)
print(vol)
print(lower)
print(upper)
np.savetxt(f'rHestonIE mutiple time scales, vol.txt', vol, delimiter=',', header=f'time: {toc - tic}')
np.savetxt(f'rHestonIE multiple time scales, lower.txt', lower, delimiter=',', header=f'time: {toc - tic}')
np.savetxt(f'rHestonIE multiple time scales, upper.txt', upper, delimiter=',', header=f'time: {toc - tic}')
time.sleep(360000)

for N_time in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    K = np.exp(-1.1 + 0.01 * np.arange(171))
    N = 12
    tic = time.perf_counter()
    vol, lower, upper = ie.call(K, N=N, N_time=N_time, m=1000000, bounce_vol=False)
    toc = time.perf_counter()
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, vol.txt', vol, delimiter=',', header=f'time: {toc - tic}')
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, lower.txt', lower, delimiter=',', header=f'time: {toc - tic}')
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, upper.txt', upper, delimiter=',', header=f'time: {toc - tic}')

'''
S, V, V_comp = ie.get_sample_path(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6,
                                  bounce_vol=True)
plt.plot(np.linspace(0, 1, 1001), S, label="stock price")
plt.plot(np.linspace(0, 1, 1001), V, label="volatility")
plt.plot(np.linspace(0, 1, 1001), V_comp[-1, :], label="low mean reversion")
plt.plot(np.linspace(0, 1, 1001), V_comp[-2, :], label="high mean reversion")
plt.legend(loc="upper left")
plt.show()
time.sleep(3600)

print("Hello World!")
Data.rHeston_smiles_precise()
Data.plot_rBergomi_smiles()
time.sleep(3600)

K = np.exp(-1.3 + 0.01 * np.arange(161))
print("True rough Heston:")
tic = time.perf_counter()
true_heston = rHeston.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=0.01,
                                         N_Riccati=3000, N_fourier=10000, L=50.)
toc = time.perf_counter()
print(true_heston)
print(f"Generating the true smile took {toc - tic} seconds.")

for N in [1, 2, 3, 4, 5, 6, 7, 8]:
    print(f"Approximation with {N} nodes, our scheme:")
    tic = time.perf_counter()
    approximated_heston = rHestonMarkov.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02,
                                                       T=0.01, N=N, N_Riccati=3000, N_fourier=10000, L=50.)
    toc = time.perf_counter()
    print(approximated_heston)
    print(f"Generating the approximated smile with N={N} took {toc - tic} seconds.")
'''
