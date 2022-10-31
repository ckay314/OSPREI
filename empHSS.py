import numpy as np
import matplotlib.pyplot as plt

def getxmodel(CHarea, SIRx):
    # CH area is in 1e8, SIRx in AU
    # first step is to get the coeffs for this CHarea
    logA = np.log(CHarea)
    mb6 = [-7.71145558e-05*logA**2 + 9.32078086e-04*logA + 6.38833448e-03, -0.01463295*logA**2 + 0.09623075*logA - 0.61998766 ]
    mb5 = [-0.00014296*logA**2 + 0.00244632*logA + 0.00320931, -0.02845527*logA**2 + 0.17027879*logA - 0.70797915]    
    mb4 = [-0.00013595*logA**2 + 0.00206428*logA + 0.00500242, 0.04222757*logA**2 -0.41331637*logA + 0.50552062]
    mb3 = [2.35626637e-05*logA**3 +-5.13349019e-04*logA**2 + 3.96532998e-03*logA + 1.72576385e-03, 0.00416472*logA**3 -0.03769825*logA**2 + 0.08601642*logA -0.41574267]    
    mb2 = [-9.10947107e-05*logA**2 + 1.39963671e-03*logA + 6.50932916e-03, 0.03394184*logA**2 -0.30485237*logA + 0.33187675]
    mb1 = [6.08428135e-05*logA**2 -3.79313672e-03*logA + 2.60322855e-02, -7.77868374e-04*logA**2 + 2.55240610e-01*logA -1.29872025e+00]
    a1 = 2.35517161e-06*logA**2 -8.73834240e-06*logA -7.24053340e-06

    # get t from SIRx and mb1
    t = (SIRx - mb3[1]) / mb3[0]

    # get full HSS position at this time
    xs = np.array([mb6[0]*t + mb6[1], mb5[0]*t + mb5[1], mb4[0]*t + mb4[1], mb3[0]*t + mb3[1], mb2[0]*t + mb2[1], a1*t**2 + mb1[0]*t + mb1[1]])
    vs = np.array([mb6[0], mb5[0], mb4[0], mb3[0], mb2[0], mb1[0]])
    return t, xs.reshape(6), vs.reshape(6), a1

def getnmodel(CHarea):
    # Area in 1e8
    logA = np.log(CHarea)
    
    y1_c2 = -0.008957771496561889*logA**3 + 0.14788075065368908*logA**2 + -0.6892846961442877*logA  + 0.9936159471088057
    y1_c1  =0.01064969310067672*logA**3 + -0.19089152424613975*logA**2 + 1.0623096741041034*logA  + -1.7703531722157
    y1_c0 = -0.0031940685737508073*logA**3 + 0.05854654547362207*logA**2 + -0.330136293302421*logA  + 1.5454902518518296
    f_y1 = lambda x: y1_c2 * x**2 + y1_c1 * x + y1_c0

    y1B_c2 = -0.04413371*logA**2 + 0.82220707*logA + -2.25053871
    y1B_c1 = -0.01282311*logA**2 + 0.06801281*logA + -0.01427758    
    y1B_c0 =  0.00986939*logA**2 + -0.07491338*logA + 1.11690102    
    f_y1B = lambda x: y1B_c2 * x**2 + y1B_c1 * x + y1B_c0

    y2A_c2 = -0.01496546*logA**3 + 0.2707287*logA**2 + -1.31857217*logA + 2.11202717
    y2A_c1 = 0.01845648*logA**3 + -0.40385558*logA**2 + 2.73747058*logA + -5.32739949    
    y2A_c0 = -0.00889893*logA**3 + 0.1905233*logA**2 + -1.2876569*logA + 3.45769381    
    f_y2A = lambda x: y2A_c2 * x**2 + y2A_c1 * x + y2A_c0

    y4_c2 = -0.01497406*logA**3 + 0.28271091*logA**2 + -1.59002866*logA + 2.92997889
    y4_c1 = 0.00591103*logA**3 + -0.13404619*logA**2 + 0.87718654*logA + -1.7548906    
    y4_c0 = -0.01601204*logA**3 + 0.32296679*logA**2 + -2.11499478*logA + 4.816763    
    f_y4 = lambda x: y4_c2 * x**2 + y4_c1 * x + y4_c0
    
    y5_c3 = -0.13408745*logA**3 + 2.24175003*logA**2 + -11.88794351*logA + 19.82889694
    y5_c2 = 0.43459931*logA**3 + -7.31087704*logA**2 + 39.13111966*logA + -66.29192212
    y5_c1 = -0.44043214*logA**3 + 7.43865698*logA**2 + -40.15995274*logA + 69.11942088    
    y5_c0 = 0.14177135*logA**3 + -2.34370086*logA**2 + 12.36000078*logA + -20.38465648    
    f_y5 = lambda x:  y5_c3 *x**3 + y5_c2 * x**2 + y5_c1 * x + y5_c0
        
    y6_c3 = -0.0817832*logA**3 + 1.36196115*logA**2 + -7.1245813*logA + 12.13843027
    y6_c2 = 0.23014318*logA**3 + -3.99397559*logA**2 + 21.55610493*logA + -37.41676283
    y6_c1 = -0.16022894*logA**3 + 3.01223949*logA**2 + -17.11065302*logA + 30.60227703    
    y6_c0 = 0.01159698*logA**3 + -0.34911121*logA**2 + 2.35176092*logA + -3.62814839    
    f_y6 = lambda x:  y6_c3 *x**3 + y6_c2 * x**2 + y6_c1 * x + y6_c0
    
    return [f_y1, f_y1B, f_y2A, f_y4, f_y5, f_y6]

def getvmodel(CHarea):
    # Area in 1e8
    logA = np.log(CHarea)
    
    y2_c2 =-0.001955824613559088*logA**3 + 0.03294930508866777*logA**2 + -0.17224209647256383*logA  + 0.2843897080757489
    y2_c1 = 0.0022094372408040077*logA**3 + -0.041883891375931556*logA**2 + 0.26848621280842966*logA  + -0.5012691474148384
    y2_c0 = 0.00016131819561611811*logA**3 + -0.0032018201178017794*logA**2 + 0.01937202011340992*logA  + 0.9717841437174127
    f_y2 = lambda x: y2_c2 * x**2 + y2_c1 * x + y2_c0
    
    y4_c2 = -0.023615794721066646*logA**3 + 0.4232470153829007*logA**2 + -2.520145550591805*logA  + 4.823510679622295
    y4_c1 = 0.023683507044893852*logA**3 + -0.42651331292286626*logA**2 + 2.5994369492129397*logA  + -5.3109371155846095
    y4_c0 = 0.01832798427972677*logA**3 + -0.37316453522753334*logA**2 + 2.474502762016573*logA  + -3.5085521838112887
    f_y4 = lambda x: y4_c2 * x**2 + y4_c1 * x + y4_c0
    
    y45_c2 = -0.03404503285151077*logA**3 + 0.6625457519412895*logA**2 + -4.155238105511233*logA  + 8.237334331217264
    y45_c1 =0.011490561422224065*logA**3 + -0.30227018561806374*logA**2 + 2.392529826235545*logA  + -5.757005561156398
    y45_c0 =0.030292311475089935*logA**3 + -0.5524570436144677*logA**2 + 3.3285515623882764*logA  + -4.782348850914333
    f_y45 = lambda x: y45_c2 * x**2 + y45_c1 * x + y45_c0
    
    y5_c2 = -0.00019685243137816135*logA**3 + 0.11250904974693461*logA**2 + -1.0261089818054094*logA  + 2.198202124794535
    y5_c1 = -0.06485590205790588*logA**3 + 0.8477425532007441*logA**2 + -3.6793005352781956*logA  + 5.24631236139184
    y5_c0 = 0.08900724766958143*logA**3 + -1.4241916890836526*logA**2 + 7.717678580444673*logA  + -12.236671760159409
    f_y5 = lambda x: y5_c2 * x**2 + y5_c1 * x + y5_c0
    
    y6_c2 = 0.010667633684460682*logA**3 + -0.17305311811374613*logA**2 + 0.8991337043131447*logA  + -1.5139656732044178
    y6_c1 = -0.028527729012572764*logA**3 + 0.47169666215470557*logA**2 + -2.503367909304984*logA  + 4.287289122538624
    y6_c0 = 0.01830195689468845*logA**3 + -0.3094010956337201*logA**2 + 1.6903979150674224*logA  + -1.9550028616927966  
    f_y6 = lambda x: y6_c2 * x**2 + y6_c1 * x + y6_c0
    
    return [f_y2, f_y4, f_y45, f_y5, f_y6]
    
def getBrmodel(CHarea):
    # Area in 1e8
    logA = np.log(CHarea)
        
    y1_c2 = -0.011094018665822861*logA**3 + 0.18170484853087907*logA**2 + -0.850850237630331*logA  + 1.2628788468384147
    y1_c1 = 0.01344513588180338*logA**3 + -0.2375932780535195*logA**2 + 1.3148483733314225*logA  + -2.2520248766868076
    y1_c0 = -0.004238986265537189*logA**3 + 0.07621976208071977*logA**2 + -0.4291557682879063*logA  + 1.7469044776619838
    f_y1 = lambda x: y1_c2 * x**2 + y1_c1 * x + y1_c0
    
    y1B_c2 = -0.028407659844596443*logA**3 + 0.4586949946894685*logA**2 + -1.9451374934938084*logA  + 2.6028102541545874
    y1B_c1 = 0.029283421684364777*logA**3 + -0.5451220562263498*logA**2 + 3.1196801152585443*logA  + -5.506675848902871
    y1B_c0 = -0.009562204126875999*logA**3 + 0.18391659715606742*logA**2 + -1.0804042757591372*logA  + 2.9561370305808343
    f_y1B = lambda x: y1B_c2 * x**2 + y1B_c1 * x + y1B_c0
    
    y3_c2 = -0.06025602152344792*logA**3 + 1.0449576860976657*logA**2 + -5.494252205252278*logA  + 9.40232502245947
    y3_c1 = 0.06165188650308154*logA**3 + -1.1450756188312363*logA**2 + 6.856969429121375*logA  + -12.78278799006528
    y3_c0 = -0.013859257500928732*logA**3 + 0.2633314418013537*logA**2 + -1.6047652295887886*logA  + 3.977506978946569
    f_y3 = lambda x: y3_c2 * x**2 + y3_c1 * x + y3_c0
    
    y4_c2 = -0.0675579023392643*logA**3 + 1.1919530566420458*logA**2 + -6.35260890379542*logA  + 10.952345698606797
    y4_c1 = 0.054092317748897924*logA**3 + -1.0256942648672225*logA**2 + 6.148911879139962*logA  + -11.895945387958824
    y4_c0 = -0.012388480212881917*logA**3 + 0.2392726605286632*logA**2 + -1.4463846067373343*logA  + 3.7148766677629577
    f_y4 = lambda x: y4_c2 * x**2 + y4_c1 * x + y4_c0

    y4A_c2 = -0.034776347093663486*logA**3 + 0.7889094302611889*logA**2 + -5.465862777461184*logA  + 11.821492075921249
    y4A_c1 = -0.022155136961942462*logA**3 + 0.14413590104335475*logA**2 + 0.9015215129306267*logA  + -5.414654119365979
    y4A_c0 = 0.010855471373291448*logA**3 + -0.11994390359809158*logA**2 + 0.1801541110365234*logA  + 1.717376036883411
    f_y4A = lambda x: y4A_c2 * x**2 + y4A_c1 * x + y4A_c0
    
    y5A_c3 = -0.19170495344954755*logA**3 + 3.1541330738630635*logA**2 + -16.515540610300448*logA  + 28.200165178456842
    y5A_c2 = 0.642393173159292*logA**3 + -10.822277172851779*logA**2 + 57.934745693832895*logA  + -100.23078132429866
    y5A_c1 = -0.6441688408956215*logA**3 + 11.12578616922947*logA**2 + -61.022680897608566*logA  + 107.23760695081374
    y5A_c0 = 0.19825972147364732*logA**3 + -3.466919482553341*logA**2 + 19.287016211031972*logA  + -33.380555466289984
    f_y5A = lambda x:  y5A_c3 * x**3 + y5A_c2 * x**2 + y5A_c1 * x + y5A_c0

    y6_c3 = -0.12624959656881116*logA**3 + 2.0907832142632086*logA**2 + -11.091801048447142*logA  + 19.229405743143747
    y6_c2 = 0.4059582902921325*logA**3 + -6.842130364302654*logA**2 + 36.85197036192945*logA  + -64.395016883432
    y6_c1 = -0.4071782238839127*logA**3 + 6.984069349709175*logA**2 + -38.2369973049835*logA  + 67.44815500713192
    y6_c0 = 0.13215013657228653*logA**3 + -2.28288896801641*logA**2 + 12.599596506287774*logA  + -21.379729076026326
    f_y6 = lambda x:  y6_c3 * x**3 + y6_c2 * x**2 + y6_c1 * x + y6_c0
    
    return [f_y1, f_y1B, f_y3, f_y4, f_y4A, f_y5A, f_y6]

def getBlonmodel(CHarea):
    # Area in 1e8
    logA = np.log(CHarea)
    
    y1_c2 = -0.004512364955742943*logA**3 + 0.05421183893149402*logA**2 + -0.051099934687997825*logA  + -0.3928066556237874
    y1_c1 = 0.004309934733892273*logA**3 + -0.0591800546437434*logA**2 + 0.16384722240873442*logA  + 0.21460989570718764
    y1_c0 = -0.0008065661309411233*logA**3 + 0.011504227922331318*logA**2 + -0.021440074610073974*logA  + 0.8592556776584489
    f_y1 = lambda x: y1_c2 * x**2 + y1_c1 * x + y1_c0    
    
    y23_c2 = -0.03223052428374426*logA**3 + 0.5662391399675146*logA**2 + -2.899818990556716*logA  + 4.862721389292354
    y23_c1 = 0.04232533845909956*logA**3 + -0.8092434905008594*logA**2 + 4.893448008880806*logA  + -9.088569822713092
    y23_c0 = -0.016184688032576763*logA**3 + 0.3173373256263623*logA**2 + -1.9958597151362512*logA  + 4.748612363975436
    f_y23 = lambda x: y23_c2 * x**2 + y23_c1 * x + y23_c0    

    y4_c2 = -0.045186506928150655*logA**3 + 0.7752840996862029*logA**2 + -3.956174785076162*logA  + 6.644216513183613
    y4_c1 = 0.04104284639535416*logA**3 + -0.746952666324115*logA**2 + 4.257771315645475*logA  + -7.960414623020648
    y4_c0 = -0.015759167251921447*logA**3 + 0.29917321547888764*logA**2 + -1.824584540344539*logA  + 4.335304721027066
    f_y4 = lambda x: y4_c2 * x**2 + y4_c1 * x + y4_c0    

    y4A_c2 = -0.010570951647483255*logA**3 + 0.29057135418172436*logA**2 + -2.236533674316334*logA  + 5.420987812195475
    y4A_c1 = -0.02528741431440361*logA**3 + 0.29849902595674366*logA**2 + -0.6520313306322177*logA  + -1.5270843603151236
    y4A_c0 = 0.005466180899419257*logA**3 + -0.03636331253247196*logA**2 + -0.25490417166346535*logA  + 2.3689270154171624
    f_y4A = lambda x: y4A_c2 * x**2 + y4A_c1 * x + y4A_c0    

    y5A_c3 = -0.3811256349985748*logA**3 + 6.247179860595188*logA**2 + -33.017403459812904*logA  + 56.7817691722361
    y5A_c2 = 1.4097575388990733*logA**3 + -23.298642805238995*logA**2 + 124.23956369469519*logA  + -214.86506293099714
    y5A_c1 = -1.6814520210055937*logA**3 + 27.926869832628903*logA**2 + -149.94317606179482*logA  + 260.613130835314
    y5A_c0 = 0.6612335459318136*logA**3 + -10.94567586886796*logA**2 + 58.690539351807786*logA  + -101.16057166747369
    f_y5A = lambda x: y5A_c3 * x**3 + y5A_c2 * x**2 + y5A_c1 * x + y5A_c0    
    
    y6_c3 = -0.25356509425373513*logA**3 + 4.146070903292366*logA**2 + -21.92007729362211*logA  + 37.780706514493936
    y6_c2 =0.9118438672050329*logA**3 + -14.985324107742207*logA**2 + 79.64047926758556*logA  + -137.56708101729322
    y6_c1 =-1.0811596493725841*logA**3 + 17.813120266953337*logA**2 + -95.01718310198146*logA  + 164.43968295424017
    y6_c0 =0.4302738997931089*logA**3 + -7.072608025266145*logA**2 + 37.66681116690262*logA  + -64.17872060903395
    f_y6 = lambda x: y6_c3 * x**3 + y6_c2 * x**2 + y6_c1 * x + y6_c0    
    
    return [f_y1, f_y23, f_y4, f_y4A, f_y5A, f_y6]

def getTmodel(CHarea):
    # Area in 1e8
    logA = np.log(CHarea)
    
    y2_c2 = 0.02054700185636421*logA**3 + -0.3046180793197968*logA**2 + 1.643317916446339*logA  + -2.7060070255325197
    y2_c1 = -0.000678218387359419*logA**3 + -0.16759312571721163*logA**2 + 1.9757017232268625*logA  + -5.076562405985274
    y2_c0 = 9.970941638576582e-05*logA**3 + 0.04922103458785567*logA**2 + -0.5524330329075601*logA  + 2.4888148321739516
    f_y2 = lambda x: y2_c2 * x**2 + y2_c1 * x + y2_c0    

    ymax_c2 = 0.14102107325353044*logA**3 + -2.70096558569321*logA**2 + 17.117539117257138*logA  + -32.8803574090646
    ymax_c1 = -0.031784797319944935*logA**3 + 0.40990724156773023*logA**2 + -1.1771619634282389*logA  + -1.0512705038228556
    ymax_c0 = 0.07985715591872361*logA**3 + -1.4569350037130706*logA**2 + 8.665605895784664*logA  + -13.20162111485989
    f_ymax = lambda x: ymax_c2 * x**2 + ymax_c1 * x + ymax_c0    
    
    y4A_c2 = -0.21821718320181485*logA**3 + 3.709333321858532*logA**2 + -19.9536467945653*logA  + 32.96563147711116
    y4A_c1 = 0.16377250548623262*logA**3 + -2.6651813963909823*logA**2 + 13.690828314699884*logA  + -20.520954389293756
    y4A_c0 = 0.10420069050901594*logA**3 + -2.125467475521785*logA**2 + 14.171308398247335*logA  + -27.927056622850756
    f_y4A = lambda x: y4A_c2 * x**2 + y4A_c1 * x + y4A_c0    
    
    y5_c2 = -0.66419232*logA**3 + 12.13082147*logA**2 + -71.30462311*logA + 132.86961
    y5_c1 = 1.15144601*logA**3 + -20.97030088*logA**2 + 122.69507118*logA -228.32579974 
    y5_c0 = -0.3250209*logA**3 +  5.96989732*logA**2 -34.74271723*logA + 66.9888603
    f_y5 = lambda x:  y5_c2 * x**2 + y5_c1*x + y5_c0 
    
    return [f_y2, f_ymax, f_y4A, f_y5]
    
def getvlonmodel(CHarea):
    # Area in 1e8
    logA = np.log(CHarea)
    
    y1_c2 = -0.396548465654734*logA**3 + 7.939494194410263*logA**2 + -51.02237880875509*logA  + 104.9167150361935
    y1_c1 = 0.7216895958629156*logA**3 + -14.59888342245146*logA**2 + 97.97713862833048*logA  + -209.08322563116678
    y1_c0 = -0.3464680590783203*logA**3 + 6.536490600532153*logA**2 + -41.306608898855*logA  + 88.60825616926617
    f_y1 = lambda x: y1_c2 * x**2 + y1_c1 * x + y1_c0    

    y2_c2 = -0.38314411879907445*logA**3 + 10.308839607355752*logA**2 + -78.66567396508007*logA  + 161.14129501646264
    y2_c1 = 1.214488947740174*logA**3 + -28.952100483609893*logA**2 + 214.77238829235196*logA  + -440.9643134553133
    y2_c0 = -0.4106565479364264*logA**3 + 8.886466477122346*logA**2 + -58.27634187492408*logA  + 118.41155900023365
    f_y2 = lambda x: y2_c2 * x**2 + y2_c1 * x + y2_c0  
    
    y4_c2 = 2.963902265853184*logA**3 + -51.42481861493056*logA**2 + 286.00980934692507*logA  + -481.67964092803356
    y4_c1 = -9.239444086741333*logA**3 + 171.54630871777763*logA**2 + -1035.574131655643*logA  + 1900.1756794418573
    y4_c0 = 1.8970608242790388*logA**3 + -34.43803317972434*logA**2 + 203.53081863027955*logA  + -380.09928874992875  
    f_y4 = lambda x: y4_c2 * x**2 + y4_c1 * x + y4_c0
    
    y4A_c2 = 10.173456346098725*logA**3 + -183.4902528792842*logA**2 + 1063.7383490190932*logA  + -1907.845481748611
    y4A_c1 = -17.49530153985073*logA**3 + 321.39422821760985*logA**2 + -1905.4364398421558*logA  + 3486.9053603138696
    y4A_c0 = 4.182484786406935*logA**3 + -74.64934029400203*logA**2 + 427.5827519387966*logA  + -753.2927545686838
    f_y4A = lambda x: y4A_c2 * x**2 + y4A_c1 * x + y4A_c0    
    
    y5_c2 = 3.2248865299367613*logA**3 + -69.41960924971988*logA**2 + 459.6512825087757*logA  + -851.3621630230559
    y5_c1 = -3.481494640889597*logA**3 + 79.26962918759791*logA**2 + -556.7420586439191*logA  + 1029.2360996803045
    y5_c0 = -2.8507138008472053*logA**3 + 49.87223574050671*logA**2 + -265.44476258804644*logA  + 474.0157578890943
    f_y5 = lambda x: y5_c2 * x**2 + y5_c1 * x + y5_c0
    
    return [f_y1, f_y2, f_y4, f_y4A, f_y5]


def getHSSprops(x_in, t_in, t0, x0, v_x, a1, funcs_in, returnReg=False):    
    n_funcsNEW = funcs_in[0]
    v_funcsNEW = funcs_in[1]
    Br_funcsNEW = funcs_in[2]
    BL_funcsNEW = funcs_in[3]
    tem_funcsNEW = funcs_in[4]
    vL_funcsNEW = funcs_in[5]
    # determine region bounds at current time (times in hours)
    bounds = x0 + v_x * (t_in - t0)
    bounds[5] += a1 * (t_in**2 - t0**2)
    if (bounds[0]-bounds[1] > x_in) and (bounds[1] > bounds[0]):
        if returnReg:
            return [1., 1., 1., 1., 1., 0.], 0
        else:
            return [1., 1., 1., 1., 1., 0.]

    if (bounds[3]<1/215.):
        if returnReg:
            return [1., 1., 1., 1., 1., 0.], 6
        else:
            return [1., 1., 1., 1., 1., 0.]

    # xs can get out of order for large enough t bc vs do not 
    # all increase from 6 to 1
    diffs = bounds[1:] - bounds[:-1]
    toFix = np.where(diffs < 0.01)[0]
    if len(toFix) != 0:
        mintoFix = toFix[0]+1
        bounds[toFix+1] = bounds[mintoFix-1] + 0.01*(toFix - mintoFix + 2)
    
    x6, x5, x4, x3, x2, x1 = bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
    #print (bounds)
    # derived xs
    x5A = 0.5 * (x6 + x5)
    
    x4A = x4 - (x2 - x4)
    x4B = x5 + (x5-x5A)
    
    x2A = 0.5 * (x3+x2)
    x1A = x2 + (x2 - x3)
    x1B = x2 + (x2 - x2A)
        
    # Figure out which region we're in and calc properties accordingly
    if x_in > bounds[0]:
        reg = np.max(np.where(bounds < x_in)) + 1
    else:
        reg = 0

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # Region Down - Ambient behind HISS
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    if reg == 0:
        
        # ------ DENSITY ------ 
        nmin = n_funcsNEW[4](x3)
        nmin = np.max([nmin,0.01])
        n6 = n_funcsNEW[5](x3)
        n6 = np.max([n6, 0.01])
        n6 = np.min([n6, 1.])      
        nmid = 0.5*(1 + nmin)
        if ((nmid -n6)/(1 - nmid) < 1) and (x5 < 1.2):
            sinterm = 2*np.arcsin((nmid -n6)/(1 - nmid))
            xmid = (sinterm*x5 - np.pi * x6) / (sinterm - np.pi)
            xstart = xmid - (x5 - xmid)
            if (x_in > xstart):
                n =nmid + (1-nmid) * np.sin(np.pi * (xmid -x_in)/(x5 - xstart))
            else:
                n = 1.
        elif (x5 >= 1.2) or (n6 > nmin):
            n = nmin + (n6 - nmin) * (x_in - x5) / (x6 - x5)
            if n > 1: n =1.
        else:
            n=1.
          
            
        # ------ RADIAL VELOCITY ------ 
        v6 = v_funcsNEW[4](x3)
        v6 = np.max([v6, 1.])      
        v5 = v_funcsNEW[3](x3)
        vmax = v_funcsNEW[2](x3)
        delv = (vmax-1)/2
        vmid = 1 + delv
        if (v5 < vmax) and (v5 > v6):
            # find where v5 falls in full sine wave
            asinR = np.arcsin((v5-vmid) / delv)
            # check if left side is short of full sine
            if v6 > 1:  
                # find where v6 falls in full sine wave
                asinL = np.arcsin((vmid-v6)/delv) 
            else:
                # could just return v =1 for reg 0 but use same code here and 
                # in reg 1
                asinL = np.pi/2
            # calc full length (half period)
            thislen = 0.5*asinL/(np.pi/2) + 0.5*asinR/(np.pi/2)
            actlen = (x5-x6)/thislen
            lendiff = actlen - (x5 - x6)
            # find actual start to sine wave
            x61 = x6 - (1-asinL/(np.pi/2)) / ( 1-asinL/(np.pi/2) + 1-asinR/(np.pi/2)) * lendiff
            # actual middle
            xmid = x61 + 0.5*actlen
            # half period
            dx = (xmid-x61) * 2
            # check within first half period of wave starting at x61
            if (x_in <= (xmid+0.5*dx)) and (x_in >= (xmid-0.5*dx)):
                v = vmid + delv * np.sin((x_in-xmid)/dx * np.pi)
            elif x_in <= (xmid-0.5*dx):
                v = 1
            else:
                v = vmax
        elif (v5 > v6) and (v5 > vmax):
            v = v6 + (v5 - v6) / (x5 - x6) * (x_in - x6)
            if v < 1: v = 1
        # set at default otherwise
        else:
            v = 1
        
        
        # ------ RADIAL B ------ 
        Br6 = Br_funcsNEW[6](x3)
        Br6 = np.min([Br6, 1])
        Br5A = Br_funcsNEW[5](x3)
        Br5A = np.max([Br5A, 0.01])
        if Br6 == 1:
            Br = 1
        elif Br5A > Br6:
            Br = 1
        else:
            c0 = 1 - Br5A
            c1 = (x5A - x6) / np.log((Br6 - 1)/(Br5A - 1))
            Br = 1 - c0 * np.exp(-(x_in-x5A)/c1)
        
        
        #  ------ LON B ------ 
        Bl6 = BL_funcsNEW[5](x3)
        Bl6 = np.min([Bl6, 1])
        Bl5A = BL_funcsNEW[4](x3)
        Bl5A = np.max([Bl5A, 0.01])
        if (Bl6 >=1): 
            Blon = 1
        elif (Bl5A < Bl6):
            c0 = 1 - Bl5A
            c1 = (x5A - x6) / np.log((Bl6 - 1)/(Bl5A - 1))
            Blon = 1 - c0 * np.exp(-(x_in-x5A)/c1)
        else:
            Blon = 1
            

        # ------ TEMPERATURE  ------ 
        tem = 1.
  
        
        # ------ LON VELOCITY  ------        
        vlon = 0.



        
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # Region E - Tail (ramp trailing HSS)    
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    elif reg == 1:
        
        # ------ DENSITY ------ 
        nmin = n_funcsNEW[4](x3)
        nmin = np.max([nmin,0.01])
        n6 = n_funcsNEW[5](x3)
        n6 = np.max([n6, 0.01])
        n6 = np.min([n6, 1.])      
        nmid = 0.5*(1 + nmin)
        if ((nmid - n6)/(1 - nmid) < 1) and (x5 < 1.2):
            sinterm = 2*np.arcsin((nmid -n6)/(1 - nmid))
            xmid = (sinterm*x5 - np.pi * x6) / (sinterm - np.pi)
            xstart = xmid - (x5 - xmid)
            if (x_in > xstart):
                n =nmid + (1-nmid) * np.sin(np.pi * (xmid -x_in)/(x5 - xstart))
            else:
                n = 1.
        elif (x5 >= 1.2) or (n6 > nmin):
            n = nmin + (n6 - nmin) * (x_in - x5) / (x6 - x5)
            if n > 1: n =1.
        else:
            n=1.


        # ------ RADIAL VELOCITY ------ 
        v6 = v_funcsNEW[4](x3)
        v6 = np.max([v6, 1.])      
        v5 = v_funcsNEW[3](x3)
        vmax = v_funcsNEW[2](x3)
        delv = (vmax-1)/2
        vmid = 1 + delv
        if (v5 < vmax) and (v5 > v6):
            # find where v5 falls in full sine wave
            asinR = np.arcsin((v5-vmid) / delv)
            # check if left side is short of full sine
            if v6 > 1:  
                # find where v6 falls in full sine wave
                asinL = np.arcsin((vmid-v6)/delv) 
            else:
                # could just return v =1 for reg 0 but use same code here and 
                # in reg 1
                asinL = np.pi/2
            # calc full length (half period)
            thislen = 0.5*asinL/(np.pi/2) + 0.5*asinR/(np.pi/2)
            actlen = (x5-x6)/thislen
            lendiff = actlen - (x5 - x6)
            # find actual start to sine wave
            xstart = x6 - (1-asinL/(np.pi/2)) / ( 1-asinL/(np.pi/2) + 1-asinR/(np.pi/2)) * lendiff
            # actual middle
            xmid = xstart + 0.5*actlen
            # half period
            dx = (xmid-xstart) * 2
            # check within first half period of wave starting at x61
            if (x_in <= (xmid+0.5*dx)) and (x_in >= (xmid-0.5*dx)):
                v = vmid + delv * np.sin((x_in-xmid)/dx * np.pi)
            elif x_in <= (xmid-0.5*dx):
                v = 1
            else:
                v = vmax
        elif (v5 > v6) and (v5 > vmax):
            v = v6 + (v5 - v6) / (x5 - x6) * (x_in - x6)
            if v < 1: v = 1
        # set at default otherwise
        else:
            if (x_in) < x5A:
                v = 1
            else:
                v = v5
        
        
                    
        # ------ RADIAL B ------ 
        if x_in < x5A:
            Br6 = Br_funcsNEW[6](x3)
            Br6 = np.min([Br6, 1])
            Br5A = Br_funcsNEW[5](x3)
            Br5A = np.max([Br5A, 0.01])
            if (Br6 - 1)/(Br5A - 1) > 0:
                c0 = 1 - Br5A
                c1 = (x5A - x6) / np.log((Br6 - 1)/(Br5A - 1))
                Br = 1 - c0 * np.exp(-(x_in-x5A)/c1)
            else:
                Br = Br6 + (Br5A - Br6) / (x5A - x6) * (x_in - x6)
        else:
            Br5A = Br_funcsNEW[5](x3)
            Br5A = np.max([Br5A, 0.01])
            Brx4A  = Br_funcsNEW[4](x3)
            Brx4 = Br_funcsNEW[3](x3)
            c0 = Brx4 - Br5A
            if ((Brx4A - Br5A) / (Brx4 - Br5A)) > 0:
                c1 = (x4A - x4) / np.log((Brx4A - Br5A) / (Brx4 - Br5A))   
                if c1 > 0: 
                    Br = Br5A + c0 * np.exp((x_in-x4)/c1)
                elif (Br5A < Brx4A):
                    Br = Br5A + (Brx4A - Br5A)/(x4A - x5A) * (x_in - x5A)
                else:
                    Br = Br5A
            elif (Br5A < Brx4A):
                Br = Br5A + (Brx4A - Br5A)/(x4A - x5A) * (x_in - x5A)
            else:
                Br = Br5A
        
        
        #  ------ LON B ------   
        if x_in < x5A:
            Bl6 = BL_funcsNEW[5](x3)
            Bl6 = np.min([Bl6, 1])
            Bl5A = BL_funcsNEW[4](x3)
            Bl5A = np.max([Bl5A, 0.01])
            if (Bl6 >= 1): 
                Blon = 1
            elif (Bl5A < Bl6):
                c0 = 1 - Bl5A
                c1 = (x5A - x6) / np.log((Bl6 - 1)/(Bl5A - 1))
                Blon = 1 - c0 * np.exp(-(x_in-x5A)/c1)
            else:
                Blon = 1
        else:
            Bl5A = BL_funcsNEW[4](x3)
            Bl5A = np.max([Bl5A, 0.01])
            Bl4A = BL_funcsNEW[3](x3)
            Bl4 = BL_funcsNEW[2](x3)
            if (Bl4A -Bl5A) / (Bl4 - Bl5A) > 0:
                c0 = Bl4 - Bl5A
                c1 = (x4A - x4) / np.log((Bl4A -Bl5A) / (Bl4 - Bl5A))
                Blon = Bl5A + c0 * np.exp((x_in - x4)/c1)
            else:
                Blon = Bl5A
        
            
        # ------ TEMPERATURE  ------ 
        temflat = tem_funcsNEW[3](x3)   
        if (temflat > 1):
            tem = 1 + (temflat - 1) / (x5 - x6) * (x_in-x6)
        else:
            tem = 1


        # ------ LON VELOCITY  ------        
        if x_in < x5A:
            vlon = 0
        else:
            vLflat = vL_funcsNEW[4](x3)
            vlon = vLflat / (x5 - x5A) * (x_in - x5A)
       
            

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # Region D - Plateau
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    elif reg == 2:
        
        # ------ DENSITY ------ 
        n2A = n_funcsNEW[2](x3)
        n1B = n_funcsNEW[1](x3)
        nmax = n2A + (n1B - n2A) / (x1B - x2A) * (x3 - x2A)
        n5 = n_funcsNEW[4](x3) 
        n4 = n_funcsNEW[3](x3) 
        if (n4 - n5) / (nmax - n5) > 0:
            c0 = nmax - n5
            c1 = (x4 - x3) / np.log((n4 - n5) / (nmax - n5))
            if c1 > 0: 
                n = n5 + c0 * np.exp((x_in-x3)/c1)
            elif (n4 > n5):
                n = n5 + (n4 - n5) / (x4 - x5) * (x_in - x5)
            else:
                n = n5
        elif (n4 > n5):
            n = n5 + (n4 - n5) / (x4 - x5) * (x_in - x5)
        else:
            n = n5
         
            
        # ------ RADIAL VELOCITY ------ 
        if (x_in < x4B):
            v6 = v_funcsNEW[4](x3)
            v6 = np.max([v6, 1.])      
            v5 = v_funcsNEW[3](x3)
            vmax = v_funcsNEW[2](x3)
            delv = (vmax-1)/2
            vmid = 1 + delv
            if (v5 < vmax) and (v5 > v6):
                # find where v5 falls in full sine wave
                asinR = np.arcsin((v5-vmid) / delv)
                # check if left side is short of full sine
                if v6 > 1:  
                    # find where v6 falls in full sine wave
                    asinL = np.arcsin((vmid-v6)/delv) 
                else:
                    # could just return v =1 for reg 0 but use same code here and 
                    # in reg 1
                    asinL = np.pi/2
                # calc full length (half period)
                thislen = 0.5*asinL/(np.pi/2) + 0.5*asinR/(np.pi/2)
                actlen = (x5-x6)/thislen
                lendiff = actlen - (x5 - x6)
                # find actual start to sine wave
                xstart = x6 - (1-asinL/(np.pi/2)) / ( 1-asinL/(np.pi/2) + 1-asinR/(np.pi/2)) * lendiff
                # actual middle
                xmid = xstart + 0.5*actlen
                # half period
                dx = (xmid-xstart) * 2
                # check within first half period of wave starting at x61
                if (x_in <= (xmid+0.5*dx)) and (x_in >= (xmid-0.5*dx)):
                    v = vmid + delv * np.sin((x_in-xmid)/dx * np.pi)
                elif x_in <= (xmid-0.5*dx):
                    v = vmax
                else:
                    v = vmax
            elif (v5 > v6) and (v5 > vmax):
                v = v6 + (v5 - v6) / (x5 - x6) * (x_in - x6)
                if v < 1: v = 1
            else:
                v = vmax
        else:
            vmax = v_funcsNEW[2](x3)
            v4 = v_funcsNEW[1](x3)
            if (v4-vmax)/(1 - vmax) > 0:
                c0 = vmax - 1
                c1 = (x4 - x2) / np.log((v4-vmax)/(1 - vmax))
                v = vmax - c0 * np.exp((x_in-x2)/c1)
            else:
                v = vmax
            
            
        # ------ RADIAL B ------ 
        Br5A = Br_funcsNEW[5](x3)
        Br5A = np.max([Br5A, 0.01])
        Brx4A  = Br_funcsNEW[4](x3)
        Brx4 = Br_funcsNEW[3](x3)
        c0 = Brx4 - Br5A
        if (((Brx4A - Br5A) / (Brx4 - Br5A)) > 0) and (x4A > 0.1):
            c1 = (x4A - x4) / np.log((Brx4A - Br5A) / (Brx4 - Br5A))   
            if c1 > 0: 
                Br = Br5A + c0 * np.exp((x_in-x4)/c1)
            else:
                Br = Br5A + (Brx4 - Br5A)/(x4 - x5A) * (x_in - x5A)
        elif (x4A < 0.1):
            Br = 1
        elif (Br5A < Brx4):
            Br = Br5A + (Brx4 - Br5A)/(x4 - x5A) * (x_in - x5A)
        else:
            Br = Br5A
        
        
        #  ------ LON B ------ 
        Bl5A = BL_funcsNEW[4](x3)
        Bl5A = np.max([Bl5A, 0.01])
        Bl4A = BL_funcsNEW[3](x3)
        Bl4 = BL_funcsNEW[2](x3)
        if ((Bl4A -Bl5A) / (Bl4 - Bl5A) > 0) and (Bl5A < Bl4A):
            c0 = Bl4 - Bl5A
            c1 = (x4A - x4) / np.log((Bl4A -Bl5A) / (Bl4 - Bl5A))
            Blon = Bl5A + c0 * np.exp((x_in - x4)/c1)
        elif (Bl5A > Bl4A):
            Blon = Bl4A + (Bl4 - Bl4A) / (x4 - x4A) * (x_in - x4A)
            if (x_in < x4A):
                Blon = Bl4A
        else:
            Blon = Bl5A
        
                         
        # ------ TEMPERATURE  ------ 
        temflat = tem_funcsNEW[3](x3)
        tem4A = tem_funcsNEW[2](x3)
        temmax = tem_funcsNEW[1](x3)
        if (((tem4A - temflat) / (temmax - temflat)) > 0) and (temflat < temmax):
            c0 = temmax - temflat
            c1 = (x4A - x4) / np.log((tem4A - temflat) / (temmax - temflat)) 
            if c1 > 0:
                tem = temflat + c0 * np.exp((x_in-x4)/c1)
            else:
                tem = temflat + (temmax - temflat) / (x4 - x4A) * (x_in - x4A)
        elif (temmax > temflat) and (x_in > x4A):
            tem = temflat + (temmax - temflat) / (x4 - x4A) * (x_in - x4A)
        
        else:
            tem  = temflat
        
        
        # ------ LON VELOCITY  ------        
        vLflat = vL_funcsNEW[4](x3)
        vL4A = vL_funcsNEW[3](x3)
        if (x_in < x4A):
            vlon = vLflat + (vL4A - vLflat) / (x4A-x5) * (x_in - x5) 
        else:                    
            v4 = vL_funcsNEW[2](x3)
            vlon = vL4A + (v4 - vL4A) / (x4 - x4A) * (x_in - x4A)


        
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------            
    # Region C - SIR part 1 (HSS side)                       
    # ----------------------------------------------------------------------------------
   # ----------------------------------------------------------------------------------
    elif reg == 3:

        # ------ DENSITY ------ 
        n2A = n_funcsNEW[2](x3)
        n1B = n_funcsNEW[1](x3)
        nmax = n2A + (n1B - n2A) / (x1B - x2A) * (x3 - x2A)
        n5 = n_funcsNEW[4](x3) 
        n4 = n_funcsNEW[3](x3) 
        if (n4 - n5) / (nmax - n5) > 0:
            c0 = nmax - n5
            c1 = (x4 - x3) / np.log((n4 - n5) / (nmax - n5))
            if c1 > 0: 
                n = n5 + c0 * np.exp((x_in-x3)/c1)
            elif (nmax > n4):
                n = n4 + (nmax - n4) / (x3 - x4) * (x_in - x4)
            else:
                n = n4
        elif (nmax > n4):
                n = n4 + (nmax - n4) / (x3 - x4) * (x_in - x4)
        else:
            n = n4
        
        
        # ------ RADIAL VELOCITY ------ 
        v4 = v_funcsNEW[1](x3)
        v2 = v_funcsNEW[0](x3)
        v2 = np.max([v2, 1])
        vmax = v_funcsNEW[2](x3)
        if (vmax > v4):
            delv = (vmax-v2)/2
            vmid = v2 + delv
            asinL = np.arcsin((v4-vmid)/delv) 
            thislen = 0.5*asinL/(np.pi/2) + 0.5
            actlen = (x2-x4)/thislen
            lendiff = actlen - (x2 - x4)
            x0Act = x4 - lendiff
            xmid = x0Act + 0.5*actlen
            dx = (xmid - x0Act) * 2
        else:
            delv = (v4-v2)/2
            vmid = v2 + delv
            xmid = 0.5 * (x4 + x2)
            dx = (xmid - x4) * 2
        v = vmid - delv * np.sin((x_in-xmid)/dx * np.pi)    
               
        
        # ------ RADIAL B ------ 
        Brx4 = Br_funcsNEW[3](x3)
        Brx3  = Br_funcsNEW[2](x3)
        if (Brx3 > 1):
            Br = Brx4 + (Brx3 - Brx4) / (x3 - x4) * (x_in - x4)
        else:
            Br = 1
        
        
        #  ------ LON B ------ 
        Bl23 = BL_funcsNEW[1](x3)
        Bl4 = BL_funcsNEW[2](x3)
        Blon = Bl4 + (Bl23 - Bl4) / (x3 - x4) * (x_in - x4)
        
        
        # ------ TEMPERATURE  ------ 
        temmax  = tem_funcsNEW[1](x3)
        tem2 = tem_funcsNEW[0](x3)
        c0 = temmax - 1
        if ((tem2 - 1) / (temmax - 1)) > 0:
            c1 = (x2 - x4) / np.log((tem2 - 1) / (temmax - 1))
            tem = 1 + c0 * np.exp((x_in-x4)/c1)
        elif (temmax > tem2):
            tem = temmax + (tem2 - temmax) / (x2 - x4) * (x_in - x4)
        else:
            tem = tem2
            

        # ------ LON VELOCITY  ------        
        vLmin = vL_funcsNEW[2](x3)
        vLmax = vL_funcsNEW[1](x3)
        amp = (vLmax - vLmin) * 0.5
        per = x2- x4
        mid = 0.5 * (vLmin + vLmax)
        vlon = mid - amp * np.cos((x_in - x4) / per * np.pi)
    
    
    
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # Region B - SIR part 2 (SW side)        
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    elif reg == 4:
        
        # ------ DENSITY ------ 
        n2A = n_funcsNEW[2](x3)
        n1B = n_funcsNEW[1](x3)
        n = n2A + (n1B - n2A) / (x1B - x2A) * (x_in - x2A)
        
        
        # ------ RADIAL VELOCITY ------ 
        v4 = v_funcsNEW[1](x3)
        v2 = v_funcsNEW[0](x3)
        v2 = np.max([v2, 1])
        vmax = v_funcsNEW[2](x3)
        if (vmax > v4):
            delv = (vmax-v2)/2
            vmid = v2 + delv
            asinL = np.arcsin((v4-vmid)/delv) 
            thislen = 0.5*asinL/(np.pi/2) + 0.5
            actlen = (x2-x4)/thislen
            lendiff = actlen - (x2 - x4)
            x0Act = x4 - lendiff
            xmid = x0Act + 0.5*actlen
            dx = (xmid - x0Act) * 2
        else:
            delv = (v4-v2)/2
            vmid = v2 + delv
            xmid = 0.5 * (x4 + x2)
            dx = (xmid - x4) * 2
        v = vmid - delv * np.sin((x_in-xmid)/dx * np.pi)
        
        
        # ------ RADIAL B ------ 
        Brx3 = Br_funcsNEW[2](x3)
        Brx3 = np.max([Brx3, 1])
        Brx1B = Br_funcsNEW[1](x3) 
        Br = Brx3 + (Brx1B - Brx3) / (x1B - x3) * (x_in - x3)
 
        #  ------ LON B ------ 
        Bl23 = BL_funcsNEW[1](x3)
        if (Bl23 < 1):
            Blon = Bl23 + (1 - Bl23)/(x1A - x3) * (x_in - x3)
        else:
            Blon = Bl23
            
        
        # ------ TEMPERATURE  ------ 
        temmax  = tem_funcsNEW[1](x3)
        tem2 = tem_funcsNEW[0](x3)
        c0 = temmax - 1
        if ((tem2 - 1) / (temmax - 1)) > 0:
            c1 = (x2 - x4) / np.log((tem2 - 1) / (temmax - 1))
            tem = 1 + c0 * np.exp((x_in-x4)/c1)
        elif (temmax > tem2):
            tem = temmax + (tem2 - temmax) / (x2 - x4) * (x_in - x4)
        else:
            tem = tem2
        
        
        # ------ LON VELOCITY  ------        
        vLmin = vL_funcsNEW[2](x3)
        vLmax = vL_funcsNEW[1](x3)
        amp = (vLmax - vLmin) * 0.5
        per = x2- x4
        mid = 0.5 * (vLmin + vLmax)
        vlon = mid - amp * np.cos((x_in - x4) / per * np.pi)
        
        
            
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # Region A - HSS reg 3, return to SW        
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    elif reg == 5:
        
        # ------ DENSITY ------ 
        if x_in < x1B:
            n2A = n_funcsNEW[2](x3)
            n1B = n_funcsNEW[1](x3)
            n = n2A + (n1B - n2A) / (x1B - x2A) * (x_in - x2A)
        else:
            n1B = n_funcsNEW[1](x3)
            n1 = n_funcsNEW[0](x3)
            if (n1 -1) / (n1B - 1) > 0:
                c0 = n1B - 1
                c1 = (x1B - x1) / np.log((n1 -1) / (n1B - 1))
                n = 1 + c0 * np.exp(-(x_in-x1B)/c1)
            else:
                n = 1.
                
        
        # ------ RADIAL VELOCITY ------ 
        v2 = v_funcsNEW[0](x3)
        v = v2 + (1 - v2) / (x1 - x2) * (x_in - x2)
        
        
        # ------ RADIAL B ------ 
        if x_in <= x1B:
            Brx3 = Br_funcsNEW[2](x3)
            Brx3 = np.max([Brx3, 1])
            Brx1B = Br_funcsNEW[1](x3) 
            Br = Brx3 + (Brx1B - Brx3) / (x1B - x3) * (x_in - x3)
        else:
            Brx1B = Br_funcsNEW[1](x3)
            Brx1 = Br_funcsNEW[0](x3)
            Brx1 = np.max([Brx1, 1])
            c0 = Brx1B - 1
            if (Brx1 -1) / (Brx1B - 1) > 0:
                c1 = (x1B - x1) / np.log((Brx1 -1) / (Brx1B - 1))
                Br = 1 + c0 * np.exp(-(x_in-x1B)/c1)
                if (x1B > x1):
                    c1 = -(0.1) / np.log((Brx1 -1) / (Brx1B - 1))
                    Br = 1 + c0 * np.exp(-(x_in-x1B)/c1)
            else:
                Br = 1.
                
        
        # ------ LON B ------ 
        if x_in < x1A:
            Bl23 = BL_funcsNEW[1](x3)
            if (Bl23 < 1):
                Blon = Bl23 + (1 - Bl23)/(x1A - x3) * (x_in - x3)
            else:
                Blon = Bl23
        else:
            # Lon B - exponential
            Bl23 = BL_funcsNEW[1](x3)
            Bl1 = BL_funcsNEW[0](x3)
            Bl1 = np.max([Bl1, 1])
            if (Bl1 -1) / (Bl23 - 1) > 0:
                c0 = Bl23 - 1
                c1 = (x1A - x1) / np.log((Bl1 -1) / (Bl23 - 1))      
                Blon = 1 + c0 * np.exp(-(x_in-x1A)/c1)
                if c1 < 0:
                    Blon = 1.
                if Blon < 1:
                    Blon = 1.
            else:
                Blon = 1.
                
            
        # ------ TEMPERATURE  ------ 
        temmax  = tem_funcsNEW[1](x3)
        tem2 = tem_funcsNEW[0](x3)
        c0 = temmax - 1
        if (tem2 - 1) / (temmax - 1) > 0:
            c1 = (x2 - x4) / np.log((tem2 - 1) / (temmax - 1))
            tem = 1 + c0 * np.exp((x_in-x4)/c1)
        else:
            tem = 1.


        # ------ LON VELOCITY  ------        
        vlonmax = vL_funcsNEW[1](x3)
        vlon1 = vL_funcsNEW[0](x3)
        c0 = vlonmax - 1
        if (vlon1 -1) / (vlonmax - 1) > 0:
            c1 = (x2 - x1) / np.log((vlon1 -1) / (vlonmax - 1)) 
            vlon = 1 + c0 * np.exp(-(x_in-x2)/c1)
        else:
            vlon = 1
  
        

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # Region Up - Upstream of HSS  
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    elif reg == 6:
        
        # ------ DENSITY ------ 
        nmax = n_funcsNEW[1](x3)
        n1 = n_funcsNEW[0](x3)
        n1 = np.max([n1, 1])
        if (n1 -1) / (nmax - 1) > 0:
            c0 = nmax - 1
            c1 = (x1B - x1) / np.log((n1 -1) / (nmax - 1))
            n = 1 + c0 * np.exp(-(x_in-x1B)/c1)
            if n < 1:
                n = 1.
            elif (x1A > x1): # can happen for large, far HSS
                c1 = -(0.1) / np.log((n1 -1) / (nmax - 1))
                n = 1 + c0 * np.exp(-(x_in-x1)/c1)
        else:
            n = 1.
            
                    
        # ------ RADIAL VELOCITY ------ 
        v = 1 
           
           
        # ------ RADIAL B ------ 
        Brx1B = Br_funcsNEW[1](x3)
        Brx1 = Br_funcsNEW[0](x3)
        Brx1 = np.max([Brx1, 1])
        c0 = Brx1B - 1
        if (Brx1 -1) / (Brx1B - 1) > 0:
            c1 = (x1B - x1) / np.log((Brx1 -1) / (Brx1B - 1))
            Br = 1 + c0 * np.exp(-(x_in-x1B)/c1)
            if (x1B > x1):
                c1 = -(0.1) / np.log((Brx1 -1) / (Brx1B - 1))
                Br = 1 + c0 * np.exp(-(x_in-x1B)/c1)
        else:
            Br = 1.
            
        
        #  ------ LON B ------ 
        Blonmax = BL_funcsNEW[1](x3)
        Bl1 = BL_funcsNEW[0](x3)
        Bl1 = np.max([Bl1, 1])
        if (Bl1 -1) / (Blonmax - 1) > 0:
            c0 = Blonmax - 1
            c1 = (x1A - x1) / np.log((Bl1 -1) / (Blonmax - 1))      
            Blon = 1 + c0 * np.exp(-(x_in-x1A)/c1)
            if c1 < 0:
                Blon = 1.
            if Blon < 1:
                Blon = 1.
        else:
            Blon = 1.
        
        # ------ TEMPERATURE  ------ 
        tem = 1.
        
        # ------ LON VELOCITY ------ 
        vlonmax = vL_funcsNEW[1](x3)
        vlon1 = vL_funcsNEW[0](x3)
        c0 = vlonmax - 1
        if (vlon1 -1) / (vlonmax - 1) > 0:
            c1 = (x2 - x1) / np.log((vlon1 -1) / (vlonmax - 1)) 
            vlon = 1 + c0 * np.exp(-(x_in-x2)/c1)
        else:
            vlon = 1.
    if returnReg:    
        return [n, v, Br, Blon, tem, vlon*1e5], reg
    else:
        return [n, v, Br, Blon, tem, vlon*1e5]
    
    
#t0, x0, v_x = getxmodel(800, 0.75)
#vfuncs = getvmodel(800)
#print (getHSSprops(1.1, t0+20, t0, x0, v_x, vfuncs))

