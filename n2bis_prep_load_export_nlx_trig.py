



################################
######## TRIGGER ########
################################



def get_trig_time_for_sujet(sujet):

    #### correct values
    if sujet == 'pat_03083_1527':

        vs_starts = [768.008963, 1070.280570]
        
        sniff_allsession = [1400.029144, 2225.759889]
        sniff_peaks = [3223,   9469,  15850,  22301,  28754,  35625,  42525, 49354,  56495,  63503,  69622,  76612,  83772,  91235, 98302, 104811, 110820, 117430, 124651, 132080, 139461, 146822, 154173, 161472, 252969, 259498, 265227, 269436, 275434, 281953, 288276, 294906, 302928, 309118, 315726, 322236, 328580, 334659, 340909, 349479, 356050, 362222, 366943, 373742, 379941, 386852, 393172, 399581, 406548]
        ac_allsession = [3100, 4550]
        ac_starts = [57640.0, 70100.0, 97410.0, 110370.0, 122770.0, 135690.0, 149600.0, 164440.0, 179540.0, 192110.0, 204690.0, 217670.0, 233220.0, 323690.0, 337530.0, 354600.0, 368610.0, 382520.0, 395640.0, 410160.0, 424110.0, 438100.0, 453690.0, 467680.0, 556230.0, 569740.0, 585320.0, 600960.0, 618940.0, 632750.0, 646370.0, 660650.0, 680040.0, 695910.0]
        ac_starts = [int(i) for i in ac_starts] 


        al_allsession = [2480, 2954]
        al_starts = [1.186e4, 9.708e4, 1.8708e5]
        al_stops = [4.312e4, 1.3547e5, 2.3087e5]

    if sujet == 'pat_03105_1551':

        vs_starts = [1624.984596, 1926.160561]
        
        sniff_allsession = [2168.62969, 2931.20041]
        sniff_peaks = [  3503,   7964,  14293,  20872,  27284,  32216,  38587, 43517,  50260,  57029,  63632,  70023,  76094,  82484, 88894,  95193, 101664, 108174, 114423, 120541, 126460, 132700, 139220, 145381, 151512, 157269, 163488, 217713, 223252, 228691, 233991, 239381, 244481, 249691, 254790, 259889, 265188, 270388, 275868, 281037, 286326, 291426, 296457, 301877, 307219, 312390, 317670, 322990, 328480, 333730, 338973, 344102, 349163, 354443, 359455, 364845, 370314, 375902]

        ac_allsession = [3550, 4850]
        ac_starts = [15960, 32575, 43049, 55500, 65750, 78299, 92700, 107200, 122200, 137299, 152400, 166549, 180400, 195000, 263950, 276800, 290250, 305000, 319500, 334099, 350250.0, 364550, 378650, 394500, 408300, 469900, 484400, 497500, 511800, 526250.0, 543500, 557199, 572949, 588050, 602099, 615599]
        ac_starts = [int(i) for i in ac_starts] 


        al_allsession = [2978, 3346]
        al_starts = [1.120e4, 7.999e4, 1.4986e5]
        al_stops = [3.335e4, 9.963e4, 1.6998e5]

    if sujet == 'pat_03128_1591':

        vs_starts = [241.577467,  543.440413]
        
        sniff_allsession = [892.9, 1662.1]
        sniff_peaks = [2105, 4776, 7575, 10784, 14269, 17868, 21468, 25047, 29267, 33162, 36495, 39492, 42377, 45283, 48326, 51286, 54639, 57916, 61427, 65027, 68869, 72603, 76180, 79738, 83516, 87494, 91382, 95099, 99285, 103150, 107071, 111289, 115111, 119025, 122706, 126564, 130379, 134494, 138349, 142173, 145974, 149655, 153356, 157084, 160561, 164422, 168250, 210774, 214004, 217522, 220815, 224005, 227425, 230959, 234330, 237249, 239884, 242804, 245623, 248763, 251214, 253833, 256689, 259595, 262651, 265661, 269158, 273444, 276714, 279757, 283177, 286674, 290813, 295319, 301087, 306575, 311920, 316796, 322415, 327261, 332405, 336911, 340869, 344887, 348689, 352921, 356845, 361358, 365787, 370140, 375043]
        ac_allsession = [2200, 3000]
        ac_starts = [41681.0, 50851.0, 60709.0, 70919.0, 81166.0, 90921.0, 102291.0, 115917.0, 129255.0, 141535.0, 153602.0, 165854.0, 178867.0, 216774.0, 229717.0, 242522.0, 255340.0, 268187.0, 281005.0, 294046.0, 306874.0, 319720.0, 332000.0, 344558.0, 356476.0, 367856.0]
        ac_starts = [int(i) for i in ac_starts] 

        al_allsession = [1787, 2067]
        al_starts = [5066, 4.150e4, 1.0109e5]
        al_stops = [29450, 68200, 135200]

    if sujet == 'pat_03138_1601':

        vs_starts = [1718.971136, 2021.155615]
        
        sniff_allsession = [2377, 3441]
        sniff_peaks = [8499, 15277, 22799, 28485, 36795, 48055, 64021, 75005, 94385, 105935, 121228, 145981, 202750, 219603, 234700, 253219, 277183, 300827, 317430, 333084, 379783, 395842, 412098, 428915, 438776, 456970, 477592, 498005, 521226]  
            
        ac_allsession = [4300, 5100]
        ac_starts = [28500.0, 47711.0, 62116.0, 76166.0, 87513.0, 101862.0, 119182.0, 136707.0, 152735.0, 168944.0, 218469.0, 236026.0, 249382.0, 260603.0, 272510.0, 284984.0, 297726.0, 310531.0, 323107.0, 336590.0, 351010.0, 351010.0, 364934.0, 381600.0]
        ac_starts = [int(i) for i in ac_starts] 

        al_allsession = [3614, 4116]
        al_starts = [24748, 105155, 200301]
        al_stops = [6.305e4, 1.5602e5, 2.3869e5]

    if sujet == 'pat_03146_1608':

        vs_starts = [550, 850]
        
        sniff_allsession = [954, 2661]
        sniff_peaks = [17116, 22015, 25989, 29638, 33517, 37303, 41206, 45277, 49301, 53083, 57256, 61460, 65589, 69875, 73884, 78111, 82526, 87456, 90797, 94707, 98860, 103002, 107393, 111640, 116113, 119758, 123447, 127588, 131549, 135640, 139824, 143915, 148775, 152877, 156925, 161172, 164868, 684862, 688027, 692114, 696213, 700264, 704580, 708914, 712991, 717425, 721967, 726487, 731837, 736435, 741490, 746388, 752237, 758017, 762541, 767188, 772029, 776445, 781687, 785959, 790449, 794870, 799998, 804466, 808657, 812934, 817432, 822904, 827315, 831519, 835857, 840338, 844672]  
            
        ac_allsession = [3050, 4250]
        ac_starts = [37560.0, 47593.0, 58886.0, 69771.0, 80877.0, 93075.0, 104979.0, 118614.0, 131370.0, 142521.0, 155029.0, 165559.0, 177277.0, 230085.0, 241201.0, 255314.0, 268434.0, 283834.0, 295668.0, 307511.0, 321988.0, 338825.0, 350623.0, 363308.0, 374574.0, 386586.0, 428066.0, 438587.0, 447937.0, 458697.0, 471294.0, 481167.0, 492078.0, 505562.0, 520855.0, 533009.0, 547229.0, 558912.0, 573628.0]
        ac_starts = [int(i) for i in ac_starts] 

        al_allsession = [2768.2, 3040.2]
        al_starts = [7.14e3, 4.707e4, 9.229e4]
        al_stops = [1.519e4, 6.024e4, 1.2382e5]

    if sujet == 'pat_03174_1634':

        vs_starts = [1062, 1362]
        
        sniff_allsession = [1880, 3150]
        sniff_peaks = [11690, 16639, 22067, 28218, 36059, 46448, 55871, 66084, 78150, 87843, 99638, 108351, 117956, 128798, 143108, 157681, 169808, 289990, 292836, 301279, 308435, 312378, 319786, 326674, 335955, 344749, 350285, 359802, 371881, 379452, 389489, 394762, 400358, 404401, 408740, 414330, 418217, 423949, 431127, 438434, 446627, 477538, 483479, 489853, 499283, 503825, 510578, 517425, 524462, 538373, 542962, 548505, 553216, 557738, 563105, 571088, 576144, 580856, 586466, 593084, 599938, 607211, 611638, 619891, 626299, 631383]  
            
        ac_allsession = [4430, 5600]
        ac_starts = [22419, 47654, 63601, 75760, 88055, 100699, 114525, 125131, 136555, 148070, 158798, 172214, 213252, 224601, 238320, 249608, 263911, 275453, 287832, 299600, 311698, 325259, 336560, 348505, 360739, 411185, 418278, 429301, 440703, 452353, 463355, 475111, 486938, 499113, 512919, 524008, 537024, 549926, 560310]
        ac_starts = [int(i) for i in ac_starts] 

        al_allsession = [3397, 4268]
        al_starts = [18303, 148594, 330040]
        al_stops = [65939, 225389, 411280]

    return vs_starts, sniff_allsession, sniff_peaks, ac_allsession, ac_starts, al_allsession, al_starts, al_stops





########################
######## ECG ########
########################


ecg_events_corrected_allsujet = {

    'pat_02459_0912': [1007705, 1010522, 1120726, 1121095, 1121458, 1121811, 1122169, 1122516, 1122863, 1123210, 1123563, 1123921, 1584942],

    'pat_02476_0929': [2.7039e5, 533976, 534261, 534540, 667162, 670598, 670944, 673689, 674002, 674332, 674661],

    'pat_02495_0949': [], # OK  

    'pat_02718_1201': [], # OK

    'pat_03083_1527': [], # OK

    'pat_03105_1551': [], # OK

    'pat_03128_1591': [], # OK

    'pat_03138_1601': [], # OK

    'pat_03146_1608': [807.477, 1005.575, 1013.808, 1029.236, 1044.353, 1060.154, 1068.274, 1094.088, 1102.949, 1111.021, 1168.914, 
    1200.529, 1201.315, 1209.257, 1217.05, 1224.98, 1233.673, 1241.987, 1251.478, 1259.322, 1267.724, 1277.123, 1283.784, 2330.470, 
    2338.375, 2346.012, 2354.411, 2362.824, 2372.09, 2380.129, 2388.978, 2407.518, 2417.976, 2426.805, 2436.726, 2446.832, 2458.523, 
    2469.778, 2488, 2507.286, 2517.201, 2525.579, 2534.69, 2543.681, 2554.103, 2571.197, 2608.240, 2617.033, 2625.784], # OK

    'pat_03174_1634': [2725.989, 2726.996, 3405.825, 3431.795, 3432.787, 3433.834, 3445.271, 3446.528, 
    3447.896, 3449.258, 3450.596, 3450.596, 3453.361, 3453.361, 3456.242, 3459.258, 3460.926, 3464.077, 3465.81, 
    3467.554, 3469.187, 3472.027, 3473.375, 3474.873, 3476.320, 3477.768, 3480.689, 3482.177, 3483.654, 3485.157, 
    3486.675, 3488.153, 3489.626, 3491.059, 3492.466, 3498.222, 3505.326, 3512.139, 3514.759, 3515.981, 3517.139, 
    3523.140, 3525.835, 3528.425, 4573.309, 4648.96, 4725.69, 4856.64, 4947.955, 4949.193, 5045.632, 5119.441, 
    5265.85, 5267.06, 5283.48, 5305.31, 5311.6, 5396.24, 5445.78, 5477.75, 5495.82, 5503.18, 5504.35, 5520.93, 5522.077, 5530.22] 

}