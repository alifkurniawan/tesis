import matplotlib.pyplot as plt

experiment_result = {
    'adam': {
        'training_30': {
            'hidden_25': {
                'drmsd_avg': [400.4654235839844, 407.8056640625, 470.0390625, 475.08245849609375, 474.65771484375,
                              468.6343688964844, 444.4172058105469, 423.5080871582031, 421.5133361816406,
                              344.94512939453125, 292.8815002441406, 258.5051574707031, 249.7576446533203,
                              235.1590118408203, 258.9526672363281, 235.77012634277344, 241.8895721435547,
                              236.8849639892578, 197.95303344726562, 171.14208984375, 166.42393493652344,
                              155.19970703125, 161.96810913085938, 158.02023315429688, 155.13584899902344,
                              166.8032684326172, 151.9856414794922, 168.69786071777344, 164.0172882080078,
                              157.52110290527344, 175.02767944335938, 155.6262969970703, 160.69593811035156,
                              170.5836181640625, 167.05078125, 192.5740509033203, 196.35926818847656],
                'validation_dataset_size': 224,
                'sample_num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                               200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
                               370],
                'train_loss_values': [1.9340084195137024, 1.8140663743019103, 1.7333406567573548, 1.6966841101646424,
                                      1.6753077387809754, 1.6436383605003357, 1.6247759819030763, 1.5841363191604614,
                                      1.5660760879516602, 1.5331052184104919, 1.5045427441596986, 1.4612240552902223,
                                      1.444921851158142, 1.410212755203247, 1.39178968667984, 1.3760053515434265,
                                      1.3403448700904845, 1.3235069155693053, 1.31842942237854, 1.2959710359573364,
                                      1.285477340221405, 1.2568637371063232, 1.2553756952285766, 1.230543839931488,
                                      1.2119799733161927, 1.203243374824524, 1.1822999119758606, 1.1652316927909852,
                                      1.148509693145752, 1.126304841041565, 1.116451048851013, 1.0954661607742309,
                                      1.077075147628784, 1.05771861076355, 1.0391275882720947, 1.0183192491531372,
                                      0.9988517761230469],
                'validation_loss_values': [291.3583170418696, 297.0408807277591, 340.781020168898, 344.35687652444375,
                                           343.77696599315317, 339.6276243366566, 323.3196514696513, 308.66007614501694,
                                           307.1380859863347, 253.32360132080038, 217.07667117937393, 191.9980876591604,
                                           186.2591927400676, 175.90264772864967, 192.63286256368025, 176.1570800328616,
                                           180.60970821394838, 177.43100409716084, 150.02621430581394,
                                           131.14742645179587, 128.77735421962456, 120.53414149303629,
                                           124.84723656139217, 121.74086508794095, 120.55094316861911,
                                           128.5782219750541, 118.14216549823243, 129.35284408375549,
                                           126.69726390658226, 121.72487534840393, 134.4042572638005,
                                           121.51046828381723, 124.08155341485636, 131.05831105550223,
                                           129.03384772865377, 146.3211184877102, 149.14335992162896]
            },
            'hidden_50': {
                'drmsd_avg': [400.4654235839844, 407.8056640625, 470.0390625, 475.08245849609375, 474.65771484375,
                              468.6343688964844, 444.4172058105469, 423.5080871582031, 421.5133361816406,
                              344.94512939453125, 292.8815002441406, 258.5051574707031, 249.7576446533203,
                              235.1590118408203, 258.9526672363281, 235.77012634277344, 241.8895721435547,
                              236.8849639892578, 197.95303344726562, 171.14208984375, 166.42393493652344,
                              155.19970703125, 161.96810913085938, 158.02023315429688, 155.13584899902344,
                              166.8032684326172, 151.9856414794922, 168.69786071777344, 164.0172882080078,
                              157.52110290527344, 175.02767944335938, 155.6262969970703, 160.69593811035156,
                              170.5836181640625, 167.05078125, 192.5740509033203, 196.35926818847656],
                'validation_dataset_size': 224,
                'sample_num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                               200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
                               370],
                'train_loss_values': [1.9340084195137024, 1.8140663743019103, 1.7333406567573548, 1.6966841101646424,
                                      1.6753077387809754, 1.6436383605003357, 1.6247759819030763, 1.5841363191604614,
                                      1.5660760879516602, 1.5331052184104919, 1.5045427441596986, 1.4612240552902223,
                                      1.444921851158142, 1.410212755203247, 1.39178968667984, 1.3760053515434265,
                                      1.3403448700904845, 1.3235069155693053, 1.31842942237854, 1.2959710359573364,
                                      1.285477340221405, 1.2568637371063232, 1.2553756952285766, 1.230543839931488,
                                      1.2119799733161927, 1.203243374824524, 1.1822999119758606, 1.1652316927909852,
                                      1.148509693145752, 1.126304841041565, 1.116451048851013, 1.0954661607742309,
                                      1.077075147628784, 1.05771861076355, 1.0391275882720947, 1.0183192491531372,
                                      0.9988517761230469],
                'validation_loss_values': [291.3583170418696, 297.0408807277591, 340.781020168898, 344.35687652444375,
                                           343.77696599315317, 339.6276243366566, 323.3196514696513, 308.66007614501694,
                                           307.1380859863347, 253.32360132080038, 217.07667117937393, 191.9980876591604,
                                           186.2591927400676, 175.90264772864967, 192.63286256368025, 176.1570800328616,
                                           180.60970821394838, 177.43100409716084, 150.02621430581394,
                                           131.14742645179587, 128.77735421962456, 120.53414149303629,
                                           124.84723656139217, 121.74086508794095, 120.55094316861911,
                                           128.5782219750541, 118.14216549823243, 129.35284408375549,
                                           126.69726390658226, 121.72487534840393, 134.4042572638005,
                                           121.51046828381723, 124.08155341485636, 131.05831105550223,
                                           129.03384772865377, 146.3211184877102, 149.14335992162896]
            },
            'hidden_125': {
                'drmsd_avg': [504.1164245605469, 510.7474060058594, 477.0649719238281, 473.41064453125,
                              450.88568115234375, 282.0032958984375, 542.3682250976562, 507.1772155761719,
                              525.9417114257812, 518.1564331054688, 408.7057800292969, 395.501220703125,
                              287.2591857910156, 267.3594055175781, 272.9082336425781, 237.4261932373047,
                              200.4827423095703, 189.7664031982422, 183.8789825439453, 201.35299682617188,
                              178.9796600341797, 172.4537811279297, 198.316162109375, 210.7042999267578,
                              205.51220703125, 223.87570190429688, 198.82473754882812, 204.24046325683594,
                              239.1134033203125, 181.60250854492188, 247.83726501464844, 242.88340759277344],
                'validation_dataset_size': 224,
                'sample_num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                               200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320],
                'train_loss_values': [1.8059015035629273, 1.6725929021835326, 1.6428096055984498, 1.6281950354576111,
                                      1.6056923747062684, 1.5912369251251222, 1.5834550261497498, 1.5592382550239563,
                                      1.5307339191436768, 1.5172709465026855, 1.4750016450881958, 1.4434085726737975,
                                      1.4026817798614502, 1.387514305114746, 1.360129964351654, 1.3396204113960266,
                                      1.3291727304458618, 1.3200640559196473, 1.3031943798065186, 1.2865205526351928,
                                      1.2768682360649108, 1.2592679738998414, 1.245306372642517, 1.2244266510009765,
                                      1.1973190188407898, 1.1873773455619812, 1.1603389024734496, 1.1433051824569702,
                                      1.1126793384552003, 1.0912015676498412, 1.0811075091362, 1.0483869194984436],
                'validation_loss_values': [363.3711964285643, 367.7422073842441, 346.7837119804732, 343.6454553070959,
                                           327.96130032769577, 213.0578873040338, 390.6667558630507, 367.5527653900725,
                                           380.4220689993499, 374.79467863820054, 296.77467549612874,
                                           287.80525557550766, 213.63985883531026, 199.80971111725398,
                                           202.2519886343728, 177.50481403458792, 151.79798546807865,
                                           144.18193647510356, 139.92824849301786, 152.00720254769746,
                                           137.42249707086202, 132.50374375248927, 150.5046431772801,
                                           159.84875401288792, 155.65946254882266, 167.67410495657958,
                                           150.76962790319837, 154.5418565465408, 177.92918517246954,
                                           138.65917137656024, 184.06813474752974, 181.47719799144647]
            },
            'hidden_250': {
                'drmsd_avg': [475.017578125, 486.3648986816406, 502.266357421875, 528.5281982421875, 457.6391906738281,
                              481.5626525878906, 497.22210693359375, 386.1423645019531, 455.8386535644531,
                              421.060791015625, 375.10235595703125, 281.2608642578125, 274.6040954589844,
                              180.05795288085938, 186.7418975830078, 163.50575256347656, 167.4431915283203,
                              167.46128845214844, 154.5019073486328, 143.60365295410156, 148.443603515625,
                              157.8722381591797, 151.03785705566406, 143.8977813720703, 150.4163818359375,
                              175.20419311523438, 185.54771423339844, 166.8349151611328, 189.7696075439453,
                              189.54981994628906], 'validation_dataset_size': 224,
                'sample_num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                               200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
                'train_loss_values': [1.769211220741272, 1.6801873445510864, 1.6599749445915222, 1.6457444310188294,
                                      1.6278255820274352, 1.6114083409309388, 1.590342378616333, 1.5559885859489442,
                                      1.533531367778778, 1.4857114195823669, 1.4508763432502747, 1.425042700767517,
                                      1.4010192632675171, 1.3949535131454467, 1.3725557327270508, 1.3593451380729675,
                                      1.3422220706939698, 1.3295957446098328, 1.3243741154670716, 1.295325469970703,
                                      1.2907560467720032, 1.264856481552124, 1.2352187633514404, 1.2203940749168396,
                                      1.2021339535713196, 1.1604907512664795, 1.1428839564323425, 1.126031267642975,
                                      1.1035578727722168, 1.0830170512199402],
                'validation_loss_values': [344.6198591896246, 351.0981634576322, 363.26853447016555, 380.962231740618,
                                           330.0822901600481, 347.112992704228, 357.9867913510178, 282.3945857522795,
                                           329.4403667810539, 305.7188596668185, 273.484211621689, 208.65940012193056,
                                           203.84588012294196, 137.4416943956748, 142.36209980620927, 125.3068095165811,
                                           128.76720166541503, 128.33173627082633, 120.2135506670729,
                                           112.17650133253933, 115.61769356216483, 122.27158154227159,
                                           117.52043412767794, 112.6338494450604, 116.88044571776803, 134.0444182418139,
                                           140.64493517972747, 128.40418254100663, 144.4067796499325, 144.1567156364386]
            },
            'hidden_500': {
                'drmsd_avg': [536.15087890625, 533.7974243164062, 462.349365234375, 373.50384521484375,
                              428.30755615234375, 473.7536926269531, 431.080078125, 337.0520324707031,
                              262.16937255859375, 256.0052185058594, 226.31307983398438, 145.41262817382812,
                              147.4139404296875, 128.95518493652344, 100.88060760498047, 98.08168029785156,
                              138.74009704589844, 133.85655212402344, 114.40283203125, 104.15373229980469,
                              122.42339324951172, 145.28440856933594, 135.5818634033203, 134.1032257080078,
                              135.3468475341797, 127.46363830566406], 'validation_dataset_size': 224,
                'sample_num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                               200, 210, 220, 230, 240, 250, 260],
                'train_loss_values': [1.7360766291618348, 1.6566931009292603, 1.654682421684265, 1.6278523445129394,
                                      1.6144542217254638, 1.584823989868164, 1.5708287119865418, 1.5141351819038391,
                                      1.477229118347168, 1.450188195705414, 1.413301372528076, 1.3969702482223512,
                                      1.3760661959648133, 1.3533055305480957, 1.3437168836593627, 1.3025951504707336,
                                      1.2771478056907655, 1.2622500658035278, 1.2295196890830993, 1.2146541237831117,
                                      1.1879598140716552, 1.1805808067321777, 1.1607080698013306, 1.1374988198280334,
                                      1.120938205718994, 1.1093169093132018],
                'validation_loss_values': [385.83790359334716, 385.6766236315593, 333.9697615489944, 274.2866100063885,
                                           311.0110031902859, 343.070533260323, 312.62666177007804, 247.91302259502723,
                                           195.5991266569637, 190.52902175954722, 169.67576779413233, 113.6659924838161,
                                           114.75596340199688, 101.79506930899579, 82.13912367622177, 80.75553887021098,
                                           108.9635736632221, 105.37803977372296, 91.26924953917187, 84.19981371254293,
                                           97.1088207608112, 113.6996327395141, 106.47030918781898, 105.11269411496735,
                                           105.86088159531634, 101.21627914433411]
            }
        }
    },
    'sgdr': {

    }

}