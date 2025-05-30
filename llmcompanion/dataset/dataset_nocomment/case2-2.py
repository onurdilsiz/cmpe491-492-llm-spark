import numpy as np
import ast
import argparse
import datetime
import math
import subprocess
from lib import tensorOps
from lib import NormalGamma
from lib import MultiplicativeWeight
from pyspark import SparkContext
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import norm
from tensorly.tenalg import khatri_rao
from numpy.linalg import solve
from tensorly.base import unfold, fold
from pyspark.accumulators import AccumulatorParam

from pyspark.mllib.linalg import DenseMatrix

from operator import add
import gc
import pyspark

sc = SparkContext(appName='CPD-MWU')
R = 5

maxIter = 500
minErrDelta = 1e-05
numExec = 500

randomSeed = 0
inputDir='/user/aggour/rpi/spark/tensor-500x500x5x10000/'
outputDir=''

regularization = 0  # None
regulParam = 0.001

sketching = 0 # No sketching
sketchingRate = 0.000001
finalSketchRate = 0.000001
errorCalcSketchingRate = 0.000001

onUpdateWeightLoop = False
mwuEpsilon = 0.15
mwuEta = 2
decompMode = 0

eye = 0

I = 0
J = 0
K = 0

A = 0
B = 0

sketchingRowsA = []
sketchingRowsB = []
sketchingRowsC = []

mabRates = []
mabArms = []

def getMSDiff(diff):
    ms = diff.days*24*60*60*1000.0
    ms += diff.seconds * 1000.0
    ms += 1.0 * diff.microseconds / 1000.0
    return ms

def getMS(start, stop):
    diff = stop - start
    return getMSDiff(diff)

def saveFactorMatrices(partition):
    ret = []
    rows = list(partition)
    error = 0.0
    for row in rows:
        label = row[0]
        Xi = row[1]
        Ki = Xi.shape[0]
	dashIdx=label.rindex('-')
	dotIdx=label.rindex('.')
	labelId=int(label[dashIdx+1:dotIdx])

	# solve for Ci
	Ci = np.zeros((Ki,R))
	ZiTZic = tensorOps.ZTZ(A, B)
	XiZic = np.dot(unfold(Xi, 0), khatri_rao([Ci, A, B], skip_matrix=0))
	if regularization > 0:
	    ZiTZic = ZiTZic + regulParam * eye
	Ci = solve(ZiTZic.T, XiZic.T).T
	#print Ci

	if outputDir!='':
	    # save Ci
	    filename = './Ci-' + str(labelId)
	    np.save(filename, Ci)

	    # save A & B
	    if labelId==0:
		filename = './A'
		np.save(filename, A)
		filename = './B'
		np.save(filename, B)

	error = error + np.square(norm(Xi - kruskal_to_tensor([Ci, A, B]), 2))

    if outputDir!='':
	subprocess.call(['hadoop fs -moveFromLocal ' + './*.npy ' + outputDir], shell=True)

    ret.append(['error',error])
    return ret

def initializeArms(n,mean):
    #print 'INITIALIZING MULTI-ARMED BANDIT ARMS...'
    global mabRates, mabArms

#    minv = 0.000000001
#    maxv = 0.000001
    minv = 0.00001
    maxv = 0.001
    if sketchingRate > 0:
	minv = sketchingRate
    if finalSketchingRate > 0:
	maxv = finalSketchingRate
    step = (maxv - minv) / (n - 1)

    for i in range(0,n):
	sketchRate = minv + i*step
	ng = NormalGamma.NG(sketchRate)
	ng.init(mean,1.0,1.0,mean)
	mabRates.append(sketchRate)
	mabArms.append(ng)
#    return

def initializeMWU(n):
    print 'INITIALIZING MULTIPLICATIVE WEIGHT UPDATES...'
    global mabRates, mabArms

#    minv = 0.000000001
#    maxv = 0.000001
    minv = 0.00001
    maxv = 0.001
    if sketchingRate > 0:
	minv = sketchingRate
    if finalSketchingRate > 0:
	maxv = finalSketchingRate
    step = (maxv - minv) / (n - 1)

    print 'MIN:',minv
    print 'MAX:',maxv
    print 'STEP:',step
    for i in range(0,n):
	sketchRate = minv + i*step
	mwu = MultiplicativeWeight.MWU(sketchRate)
	mwu.init(1.0)
	mwu.setEpsilon(mwuEpsilon)
	mwu.setEta(mwuEta)
	mabRates.append(sketchRate)
	mabArms.append(mwu)

    mwu = MultiplicativeWeight.MWU(1.0)
    mwu.init(1.0)
    mwu.setEpsilon(mwuEpsilon)
    mwu.setEta(mwuEta)
    mabRates.append(1.0)
    mabArms.append(mwu)
#    return

def getTensorDimensions(partition):
    """
    Spark job to process each slice and return its local tensor dimensions.
    """
#    print '****** get tensor dim ******'
    ret = []
    rows = list(partition)
    for row in rows:
	Xi = row[1]
	a = []
	a.extend(Xi.shape)
	a.append(np.square(norm(Xi, 2)))
	ret.append(a)
    return [tensorOps.getDim (ret)]

def initializeData(partition):
    """
    Converts binary file of tensor slice to in-memory numpy array.  Input 
    partitions are of the form (tensor_slice_filename, binary content) 
    and outputs are of the form (tensor_slice_filename, numpy array of tensor slice).
    """
#    print '******* initializing *******'
    ret = []
    rows = list(partition)
    for row in rows:
	fsav = file('tmp.npy', 'wb')
	fsav.write(row[1])
	fsav.close()
	Xi = np.load('tmp.npy')
        ret.append([row[0], Xi])
    return ret

def updateSketchingRate(sketchingRate, errDelta, step):
    # fixed schedule - multiply rate x2 ever 2 iterations
    if sketching == 4 and (step % 2) == 0:
	sketchingRate = sketchingRate * 2.0

    # fixed buckets
    elif sketching == 5:
	'''
	if 0 <= errDelta <= 0.005:
	    sketchingRate = sketchingRate * 4
	elif 0.005 < errDelta <= 0.05:
	    sketchingRate = sketchingRate * 2
	elif 0.05 < errDelta <= 0.2:
	    sketchingRate = sketchingRate * 1.5
	'''
	#if 0 <= errDelta <= 0.0001:
	if errDelta <= 0.0001:
	    sketchingRate = 0.000001
	elif 0.0001 < errDelta <= 0.001:
	    sketchingRate = 0.000000667
	elif 0.001 < errDelta <= 0.002:
	    sketchingRate = 0.000000334
	elif 0.002 < errDelta:
	    sketchingRate = 0.000000001

    # fixed function
    elif sketching == 6 and 0 <= errDelta <= 1:
	sketchingRate = 0.000001 / (1 + errDelta*1000.0)
#    print '  delta =',errDelta
#    print '  new rate =',sketchingRate

    if sketchingRate > finalSketchingRate:
	sketchingRate = finalSketchingRate

#    print '  rate =',sketchingRate
    return sketchingRate

def showArms():
    for i in range(0,len(mabArms)):
	print str(i), ':', mabRates[i], ' == ', mabArms[i].getArm()
	mabArms[i].printVals()

def updateMABPosteriorAndSelectNextArm(sketchingRate, reward, step):
    rateIndex = mabRates.index(sketchingRate)
    ng1 = mabArms[rateIndex]
    ng1.updatePosterior(reward)
    sampleVals = map(NormalGamma.NG.sample, mabArms)
    mean, variance = ng1.getEstimates()
    print step,",SR,",sketchingRate,",mean,",mean,",var,",variance
    print 'Arm options:'
    print sampleVals
    key = np.argmax(sampleVals)
    ng2 = mabArms[key]
    #showArms()
    return ng2.getArm()

def updateMWUWeightAndSelectNextArm(sketchingRate, reward):
    global onUpdateWeightLoop
    if sketchingRate <= 0.0:
	return mabArms[0].getArm()
    rateIndex = mabRates.index(sketchingRate)
    ng1 = mabArms[rateIndex]
    if onUpdateWeightLoop:
	if rateIndex >= 0:
	    #print 'updating weight for',sketchingRate,'with reward',reward
	    ng1.updateWeight(-1.0 * reward)
	    rateIndex = rateIndex + 1
	else:
	    rateIndex = 0
	if rateIndex >= len(mabRates):
	    onUpdateWeightLoop = False
	else:
	    ng2 = mabArms[rateIndex]
	    return ng2.getArm()
    # else
    getWeightVals = map(MultiplicativeWeight.MWU.getWeight, mabArms)
    print 'WEIGHTS:',getWeightVals
    weightSum = math.fsum(getWeightVals)
    randVal = np.random.rand() * weightSum
    sum = 0.0
    for key in range(0,len(getWeightVals)):
	sum = sum + getWeightVals[key]
	if randVal <= sum:
	    break
    ng2 = mabArms[key]
    return ng2.getArm()

def selectRandomRate (step):
    rateIndex = np.random.randint(0,4)
#    print step,",SR,",mabRates[rateIndex]
    return mabRates[rateIndex]

def calculateSketchingValues (sketchingRate):
    sketchingRows = math.ceil(I * J * K * sketchingRate)
    sketchingRows_square_root = int(math.ceil(math.sqrt(sketchingRows)))
    sketchingRows_P = sketchingRows
    sketchingRows_P_cube_root = int(math.ceil(sketchingRows_P**(1.0/3)))
    return sketchingRows, sketchingRows_square_root, sketchingRows_P, sketchingRows_P_cube_root

def singleModeALSstep(partition):
    """
    Runs a single step of Alternating Least Squares to solve for one of A (mode = 1),
    B (mode = 2), or C (mode = 3) matrix.
    """
    '''
    if decompMode == 1:
        print 'Solving for A....'
    elif decompMode == 2:
        print 'Solving for B....'
    elif decompMode == 3:
        print 'Solving for Ci...'
    '''
    ret = []
    rows = list(partition)
    ZiTZi = 0
    XiZi = 0

    error = 0.0

    for row in rows:
        label = row[0]
        Xi = row[1]
        Ki = Xi.shape[0]
	# make sure not to skip over slice if we're calculating error on full tensor
#	if (sketching > 0 or (decompMode == 3 and errorCalcSketchingRate < 1)) and not (decompMode == 3 and errorCalcSketchingRate == 1) and not (decompMode == 3 and onUpdateWeightLoop):
	if ((sketching > 0 and sketchingRate < 1.0) or (decompMode == 3 and errorCalcSketchingRate < 1)) and not (decompMode == 3 and errorCalcSketchingRate == 1) and not (decompMode == 3 and onUpdateWeightLoop):
	    dashIdx=label.rindex('-')
	    dotIdx=label.rindex('.')
	    labelId=int(label[dashIdx+1:dotIdx])
	    minIndex = labelId
	    maxIndex = labelId + Ki - 1
# dalia - IS THIS A PROBLEM? THIS WILL SELECT ROWS OF C WHEN CALCULATING FULL ERROR, BUT NOT SURE THESE ROWS ARE USED
	    selectRowsC = sketchingRowsC[(sketchingRowsC >= minIndex) & (sketchingRowsC <= maxIndex)]
	    selectRowsC = selectRowsC - minIndex
	    if len(selectRowsC) == 0:
		continue;

	# always solve for Ci first!
	Ci = np.zeros((Ki,R))
#	if sketching == 1 or sketching == 3:
#	if (decompMode < 3 and (sketching == 1 or sketching >= 3)) or (decompMode == 3 and 0 < errorCalcSketchingRate < 1) and not onUpdateWeightLoop:
	if (decompMode < 3 and (sketching == 1 or sketching >= 3) and sketchingRate < 1.0) or (decompMode == 3 and 0 < errorCalcSketchingRate < 1) and not onUpdateWeightLoop:
            ZiTZic = tensorOps.ZTZ(A[sketchingRowsA,:], B[sketchingRowsB,:])
            XiZic = np.dot(unfold(Xi[:,sketchingRowsA,:][:,:,sketchingRowsB], 0), khatri_rao([Ci, A[sketchingRowsA,:], B[sketchingRowsB,:]], skip_matrix=0))
	    '''
	    if (decompMode == 3):
		print 'Solving for partial C'
	    '''
	# don't need a sketching == 2, since else is the same
	else:
	    '''
	    if (decompMode == 3):
		print 'Solving for full C'
	    '''
            ZiTZic = tensorOps.ZTZ(A, B)
            XiZic = np.dot(unfold(Xi, 0), khatri_rao([Ci, A, B], skip_matrix=0))
        #ZiTZic = tensorOps.ZTZ(A, B)
        #XiZic = np.dot(unfold(Xi, 0), khatri_rao([Ci, A, B], skip_matrix=0))
        if regularization > 0:
            ZiTZic = ZiTZic + regulParam * eye
	# I don't have Ci yet...
	#if regularization == 2:
	#    XiZi = XiZi + regulParam * Ci
        Ci = solve(ZiTZic.T, XiZic.T).T
#	print 'Xi=\n',Xi
#	print 'new Ci=\n',Ci

        if decompMode == 1:
#	    if sketching == 1 or sketching >= 3:
	    if (sketching == 1 or sketching >= 3) and sketchingRate < 1.0:
                ZiTZi = ZiTZi + tensorOps.ZTZ(B[sketchingRowsB,:], Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:][:,:,sketchingRowsB], 1), khatri_rao([Ci[selectRowsC,:], A, B[sketchingRowsB,:]], skip_matrix=1))
	    elif sketching == 2:
                ZiTZi = ZiTZi + tensorOps.ZTZ(B, Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:], 1), khatri_rao([Ci[selectRowsC,:], A, B], skip_matrix=1))
	    else:
                ZiTZi = ZiTZi + tensorOps.ZTZ(B, Ci)
#                XiZi = XiZi + tensorOps.unfolded_3D_matrix_multiply(decompMode, Xi, Ci, B, I, J, Ki, R)
                XiZi = XiZi + np.dot(unfold(Xi, 1), khatri_rao([Ci, A, B], skip_matrix=1))
        elif decompMode == 2:
#	    if sketching == 1 or sketching >= 3:
	    if (sketching == 1 or sketching >= 3) and sketchingRate < 1.0:
                ZiTZi = ZiTZi + tensorOps.ZTZ(A[sketchingRowsA,:], Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:][:,sketchingRowsA,:], 2), khatri_rao([Ci[selectRowsC,:], A[sketchingRowsA,:], B], skip_matrix=2))
	    elif sketching == 2:
                ZiTZi = ZiTZi + tensorOps.ZTZ(A, Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:], 2), khatri_rao([Ci[selectRowsC,:], A, B], skip_matrix=2))
	    else:
                ZiTZi = ZiTZi + tensorOps.ZTZ(A, Ci)
#                XiZi = XiZi + tensorOps.unfolded_3D_matrix_multiply(decompMode, Xi, Ci, A, I, J, Ki, R)
                XiZi = XiZi + np.dot(unfold(Xi, 2), khatri_rao([Ci, A, B], skip_matrix=2))
        elif decompMode == 3:
#	    if sketching == 1 or sketching == 3:
	    if 0 < errorCalcSketchingRate < 1 and not onUpdateWeightLoop:
		error = error + np.square(norm(Xi[selectRowsC,:,:][:,sketchingRowsA,:][:,:,sketchingRowsB] - kruskal_to_tensor([Ci[selectRowsC,:], A[sketchingRowsA,:], B[sketchingRowsB,:]]), 2))
		#print 'Error calc with partial C'
	    elif sketching == 2:
		error = error + np.square(norm(Xi[selectRowsC,:,:] - kruskal_to_tensor([Ci[selectRowsC,:], A, B]), 2))
	    else:
		#print 'Error calc with full C'
		error = error + np.square(norm(Xi - kruskal_to_tensor([Ci, A, B]), 2))
		#print 'local error =',np.square(norm(Xi - kruskal_to_tensor([Ci, A, B]), 2))
        else:
            print 'Unknown decomposition mode. Catastrophic error. Failing now...'

    if (len(rows) > 0) and (decompMode < 3):
        ret.append(['ZTZ',ZiTZi])
        ret.append(['XZ',XiZi])
    elif (decompMode == 3):
        ret.append(['error',error])
#	print 'cumulative error =',error
    del ZiTZi, XiZi
    return ret

def rowNormCMatrix(partition):
    """
    Calculate squared row norm of C factor matrices
    """
    ret = []
    rows = list(partition)
# dalia
    for row in rows:
	label = row[0]
	Xi = row[1]
        Ki = Xi.shape[0]
	Ci = np.zeros((Ki,R))
	ZiTZic = tensorOps.ZTZ(A, B)
	XiZic = np.dot(unfold(Xi, 0), khatri_rao([Ci, A, B], skip_matrix=0))
	if regularization > 0:
	    ZiTZic = ZiTZic + regulParam * eye
	Ci = solve(ZiTZic.T, XiZic.T).T
	dashIdx=label.rindex('-')
	dotIdx=label.rindex('.')
	labelId=int(label[dashIdx+1:dotIdx])
	rowNormCi = np.square(np.linalg.norm(Ci, axis=1))
	ret.append([labelId, rowNormCi])
    return ret

def calculateFNorm(partition):
    """
    Calculate Frobenius Norm of tensor slices.
    """
    ret = []
    rows = list(partition)
    normX = 0.0
    for row in rows:
        Xi = row[1]
        normX = normX + np.square(norm(Xi, 2))
        '''
        (Ki,I,J) = Xi.shape
        for i in range(0,I):
            for j in range(0,J):
                for k in range(0,Ki):
                    normX = normX + np.square(Xi.item((k,i,j)))
        '''
    return ([normX])

def calculateError(partition):
    """
    Calculate Frobenius Norm of difference between tensor slices and decomposed tensor.
    """
    ret = []
    rows = list(partition)
    normX = 0.0
    error = 0.0
    for row in rows:
        Xi = row[1]
        Ci = row[2]
        normX = normX + np.square(norm(Xi, 2))
        error = error + np.square(norm(Xi - kruskal_to_tensor([Ci, A, B]), 2))
        '''
        (Ki,I,J) = Xi.shape
        for i in range(0,I):
            for j in range(0,J):
                for k in range(0,Ki):
                    sum = 0.0
                    for r in range(0,R):
                        sum = sum + A.item(i,r) * B.item(j,r) * Ci.item(k,r)
                    x = Xi.item((k,i,j))
                    error = error + np.square(sum) - (2.0*sum*x)
                    normX = normX + np.square(x)
        '''
    ret.append(['error',error])
    ret.append(['normX',normX])
    return ret

def selectRowsNormWeighted(mat, maxVal, count):
    rowNorm = np.square(np.linalg.norm(mat, axis=1))
    # dalia - should I square this?
    rowNormSum = sum(rowNorm)
    rowNorm = rowNorm / rowNormSum
    return np.random.choice(maxVal, count, replace=False, p=rowNorm)

def parafac_als(inputDir, outputDir, numExec, R, maxIter, minErrDelta, regularization, regulParam, sketching, randomSeed):
    """
    Run PARAFAC ALS on tensor in input directory.
    :param inputDir:
      Input directory of tensor slice files.
    :param outputDir:
      Output directory of factor matrix slice files.
    :param numExec:
      Number of Spark executors to use. If numExec is less than the number of slices, not all may be used.
    :param R:
      Number of rank-1 tensors to decompose input tensor into.
    :param maxIter:
      Stopping criteria - maximum number of iterations.
    :param minErrDelta:
      Stopping criteria - minimum error delta after which we stop.
    :param regularization:
      Type of regularization (0 = None, 1 = L2/Tikhonov, 2 = Proximal)
    :param regulParam:
      Regularization parameter (a.k.a., lambda)
    :param sketching:
      Sketching approach (0 = None, 1 = CPRAND, 2 = % rows, 3 = % of entries)
    """
    global decompMode
    global I, J, K
    global A, B
    global eye
    global sketchingRowsC
    global sketchingRowsA
    global sketchingRowsB
    global sketchingRate
    global errorCalcSketchingRate
    global mabRates, mabArms
    global onUpdateWeightLoop

    print '********************************************************************'
    print '********************************************************************'
    print '********************************************************************'
    print 'Initializing...'
    print '    Input directory:', inputDir
    if outputDir!='':
	print '    Output directory:', outputDir
    print '    Number of Spark executors:', numExec
    print '    Tensor rank:', R
    print '    Stopping criteria:'
    print '        Max iterations:',maxIter
    print '        Min error delta:',minErrDelta
    if regularization==0:
        print '    Regularization: None'
    elif regularization==1:
        print '    Regularization: L2/Tikhonov'
        print '        Regularization parameter:',regulParam
    elif regularization==2:
        print '    Regularization: Proximal'
        print '        Regularization parameter:',regulParam
    if sketching==0:
	print '    Sketching: None'
    elif sketching==1:
	print '    Sketching: CPRAND'
    elif sketching==2:
	print '    Sketching: Random slice selection'
	print '    Sketching rate:',sketchingRate
    elif sketching==3:
	print '    Sketching: Random entry selection'
	print '    Sketching rate:',sketchingRate
    elif sketching==4:
	print '    Sketching: Increase sketching rate x2 every 2 iterations'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
    elif sketching==5:
	print '    Sketching: Increase sketching rate by buckets iterations'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
    elif sketching==6:
	print '    Sketching: Increase sketching rate by function'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
    elif sketching==7:
	print '    Sketching: Multi-Armed Bandit approach'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
#	initializeArms(4,0.1)
    elif sketching==8:
	print '    Sketching: Random rate selection'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
	initializeArms(4,1)
    elif sketching==9:
	print '    Sketching: Multiplicative Weight Updates (Label Efficient Forecaster) (random entry selection)'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
	print '    Epsilon:',mwuEpsilon
	print '    Eta    :',mwuEta
	initializeMWU(4)
    elif sketching==10:
	print '    Sketching: Multiplicative Weight Updates (Label Efficient Forecaster) (row norm weighted sampling)'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
	print '    Epsilon:',mwuEpsilon
	print '    Eta    :',mwuEta
	initializeMWU(4)
    elif sketching==11:
	print '    Sketching: Row norm weighted entry selection'
	print '    Sketching rate:',sketchingRate
    if errorCalcSketchingRate > 0:
	print '    Error calc sketching rate:',errorCalcSketchingRate

    initialSketchingRate = sketchingRate

    startAll = datetime.datetime.now()
#    print 'Reading files from HDFS', datetime.datetime.now()
    rows = sc.binaryFiles(inputDir, numExec)

    # turn tensor binary files into ndarray's
#    tensorRDD = rows.mapPartitions(initializeData).persist(pyspark.StorageLevel.MEMORY_AND_DISK)
#    tensorRDD = rows.mapPartitions(initializeData).persist(pyspark.StorageLevel.DISK_ONLY)
    tensorRDD = rows.mapPartitions(initializeData).persist(pyspark.StorageLevel.MEMORY_ONLY)
    del rows
    print '    Tensor slice count =', tensorRDD.count()

    # Do a first pass to get dimensions of slices to initialize A and B
    # also get the Frobenius norm of the tensor
    print 'Getting tensor dimensions...'
#    print 'Getting tensor dimensions', datetime.datetime.now()
    dimRDD = tensorRDD.mapPartitions(getTensorDimensions).collect()
    (K,I,J,normX) = tensorOps.getDim (dimRDD)
    I = int(I)
    J = int(J)
    K = int(K)
    eye = np.identity (R)
    print '    I =',I,', J =',J,', K =',K
    print '       normX^2 =',normX

    # set random seed so I can use the same initial conditions across runs
    if randomSeed != 0:
	np.random.seed(randomSeed)

    print 'Initializing decomposition matrices...'
#    print 'Initializing decomposition matrices', datetime.datetime.now()
    # initialize A
    A = np.random.rand(I,R)
    #print 'A:\n--\n',A
    # initialize B
    B = np.random.rand(J,R)
    #print 'B:\n--\n',B

    # set max values for skewed sketching
    if sketching==1:
	sketchingRows = math.ceil(10 * R * math.log(R, 10))
	sketchingRows_square_root = int(math.ceil(math.sqrt(sketchingRows)))
	sketchingRows_P = 372.0
	sketchingRows_P_cube_root = int(math.ceil(sketchingRows_P**(1.0/3)))
	sketchingRate = sketchingRows / (I*J*K*1.0)
    elif sketching==2:
	sketchingRows = int(math.ceil(K * sketchingRate))
    elif sketching>=3:
	sketchingRows, sketchingRows_square_root, sketchingRows_P, sketchingRows_P_cube_root = calculateSketchingValues (sketchingRate)

    if 0 < errorCalcSketchingRate < 1:
	errorCalcSketchingRows = math.ceil(I * J * K * errorCalcSketchingRate)
	errorCalcSketchingRows_square_root = int(math.ceil(math.sqrt(errorCalcSketchingRows)))
	errorCalcSketchingRows_P = errorCalcSketchingRows
	errorCalcSketchingRows_P_cube_root = int(math.ceil(errorCalcSketchingRows_P**(1.0/3)))
# dalia
#	sketchingRows_IJK = (1.0 * I * J * K) / errorCalcSketchingRows_P
	sketchingRows_IJK = (4.0 * I * J * K) / errorCalcSketchingRows_P

    # initialize error values and stopping criterion thresholds
    oldError = 100
    lowestError = 100
    errorNoChangeCount = 0
    errorChangeBelowThreshold = 0
    maxItersWithoutErrorChange = 30
    maxItersWithErrorChangeBelowThreshold = 5
    if errorCalcSketchingRate > 0 :
#	maxItersWithoutErrorChange = 1.5 * round(4.0361*pow(errorCalcSketchingRate,-0.107))
	maxItersWithoutErrorChange = round(4.0361*pow(errorCalcSketchingRate,-0.107))
#	maxItersWithoutErrorChange = round(3.1533*pow(errorCalcSketchingRate,-0.11))
    errDelta = 1

    if sketching == 7:
	runningAve = 0.0
	onUpdateWeightLoop = True

    # PARAFAC Alternating Least Squares loop
    print 'Executing decomposition...'
#    print 'Executing decomposition', datetime.datetime.now()

    if sketching in (9,10) and sketchingRate == 0.0:
	print 'About to select a sketching rate arm...'
	sketchingRate = updateMWUWeightAndSelectNextArm (sketchingRate, 1.0)
	print 'Rate =',sketchingRate
	sketchingRows, sketchingRows_square_root, sketchingRows_P, sketchingRows_P_cube_root = calculateSketchingValues (sketchingRate)
    elif sketching == 8:
	sketchingRate = selectRandomRate (0)
	sketchingRows, sketchingRows_square_root, sketchingRows_P, sketchingRows_P_cube_root = calculateSketchingValues (sketchingRate)

    mapTime = 0.0
    errorTime = 0.0
    startSteps = datetime.datetime.now()
    #maxExecTime = 0.0
    for step in range(0,maxIter):
	if sketching in (7,8,9,10):
	    if sketching in (9,10) and onUpdateWeightLoop:
		A = backupA
		B = backupB
		oldError = backupError
	    mabStart = datetime.datetime.now()
#        print '--- Iteration',step
        # solve for A
        decompMode = 1
#        print 'Solving for A', datetime.datetime.now()
#	if sketching==1 or sketching>=3:
	if (sketching == 1 or sketching >= 3) and sketchingRate < 1.0:
	    numB = np.random.randint(2,min(sketchingRows_square_root, J))
	    numC = int(math.ceil(sketchingRows/numB))
	    if numC > K:
		numC = K
		numB = int(math.ceil(sketchingRows/numC))
		if numB > J:
		    numB = J
	    #print 'rate=',sketchingRate
	    #print 'numB=',numB
	    #print 'numC=',numC
	    #sketchingRowsA = np.unique(np.random.randint(0,I,numC))
	    #sketchingRowsA = np.random.choice(I,numC,replace=False)
	    sketchingRowsA = range(0,I)
	    if sketching in (10,11):
		# dalia - get weights of arms for B and C, then weighted select
		sketchingRowsB = selectRowsNormWeighted(B, J, numB)
		indexedRowNorms = tensorRDD.mapPartitions(rowNormCMatrix).sortByKey().values().collect()
		rowNormC = np.concatenate(indexedRowNorms)
		rowNormSum = sum (rowNormC)
		rowNormC = rowNormC / rowNormSum	
		sketchingRowsC = np.random.choice(K,numC,replace=False,p=rowNormC)
	    else:
		#sketchingRowsB = np.unique(np.random.randint(0,J,numB))
		sketchingRowsB = np.random.choice(J,numB,replace=False)
		#print 'B rows=',sketchingRowsB
		#sketchingRowsC = np.unique(np.random.randint(0,K,numC))
		sketchingRowsC = np.random.choice(K,numC,replace=False)
		#print 'C rows=',sketchingRowsC
	elif sketching==2:
	    sketchingRowsC = np.random.randint(0,K,sketchingRows)
	startMap = datetime.datetime.now()
        XZandZTZ = tensorRDD.mapPartitions(singleModeALSstep)
        sums = XZandZTZ.reduceByKeyLocally(add)
	mapTime = mapTime + getMS(startMap, datetime.datetime.now())
        XZ = sums['XZ']
        ZTZ = sums['ZTZ']
        if regularization > 0:
            ZTZ = ZTZ + regulParam * eye
        if regularization == 2:
            XZ = XZ + regulParam * A
#        A = np.matmul(XZ, np.linalg.inv(ZTZ))
        A = solve(ZTZ.T, XZ.T).T
#        print 'new A=\n',A
        del XZandZTZ, XZ, ZTZ

        # solve for B
        decompMode = 2
#        print 'Solving for B', datetime.datetime.now()
#	if sketching==1 or sketching>=3:
	if (sketching == 1 or sketching >= 3) and sketchingRate < 1.0:
	    numA = np.random.randint(2,min(sketchingRows_square_root, I))
	    numC = int(math.ceil(sketchingRows/numA))
	    if numC > K:
		numC = K
		numA = int(math.ceil(sketchingRows/numC))
		if numA > I:
		    numA = I
	    #print 'rate=',sketchingRate
	    #print 'numA=',numA
	    #print 'numC=',numC
	    sketchingRowsB = range(0,J)
	    if sketching in (10,11):
		# dalia - get weights of arms for A and C, then weighted select
		sketchingRowsA = selectRowsNormWeighted(A, I, numA)
		indexedRowNorms = tensorRDD.mapPartitions(rowNormCMatrix).sortByKey().values().collect()
		rowNormC = np.concatenate(indexedRowNorms)
		rowNormSum = sum (rowNormC)
		rowNormC = rowNormC / rowNormSum	
		sketchingRowsC = np.random.choice(K,numC,replace=False,p=rowNormC)
	    else:
		#sketchingRowsA = np.unique(np.random.randint(0,I,numA))
		sketchingRowsA = np.random.choice(I,numA,replace=False)
		#sketchingRowsC = np.unique(np.random.randint(0,K,numC))
		sketchingRowsC = np.random.choice(K,numC,replace=False)
	startMap = datetime.datetime.now()
        XZandZTZ = tensorRDD.mapPartitions(singleModeALSstep)
        # 'Locally' automatically creates a dict of the results
        sums = XZandZTZ.reduceByKeyLocally(add)
	mapTime = mapTime + getMS(startMap, datetime.datetime.now())
        XZ = sums['XZ']
        ZTZ = sums['ZTZ']
        if regularization > 0:
            ZTZ = ZTZ + regulParam * eye
        if regularization == 2:
            XZ = XZ + regulParam * B
#        B = np.matmul(XZ, np.linalg.inv(ZTZ))
        B = solve(ZTZ.T, XZ.T).T
#        print 'new B=\n',B
        del XZandZTZ, XZ, ZTZ

        # solve for Ci's
        decompMode = 3
#        print 'Solving for C', datetime.datetime.now()
#	if sketching==1 or sketching>=3:
	if 0 < errorCalcSketchingRate < 1 and not onUpdateWeightLoop:
	    numA = np.random.randint(2,min(errorCalcSketchingRows_P_cube_root, I))
	    numB = np.random.randint(2,min(errorCalcSketchingRows_P_cube_root, J))
	    numC = int(math.ceil(errorCalcSketchingRows_P/(numA*numB)))
	    if sketching in (10,11):
		# dalia - get weights of arms for A and B, then weighted select
		sketchingRowsA = selectRowsNormWeighted(A, I, numA)
		sketchingRowsB = selectRowsNormWeighted(B, J, numB)
	    else:
		#sketchingRowsA = np.unique(np.random.randint(0,I,numA))
		sketchingRowsA = np.random.choice(I,numA,replace=False)
		#sketchingRowsB = np.unique(np.random.randint(0,J,numB))
		sketchingRowsB = np.random.choice(J,numB,replace=False)
		#sketchingRowsC = np.unique(np.random.randint(0,K,numC))
	    sketchingRowsC = np.random.choice(K,numC,replace=False)
	startMap = datetime.datetime.now()
	errorRDD = tensorRDD.mapPartitions(singleModeALSstep)

        # calculate error
#        print 'Calculating error', datetime.datetime.now()
	sums = errorRDD.reduceByKeyLocally(add)
	mapTime = mapTime + getMS(startMap, datetime.datetime.now())
	errorTime = errorTime + getMS(startMap, datetime.datetime.now())

#	if sketching==1 or sketching>=3:
	if 0 < errorCalcSketchingRate < 1 and not onUpdateWeightLoop:
	    newError = np.sqrt((sketchingRows_IJK * sums['error']) / normX)
	elif sketching==2:
	    newError = np.sqrt(sums['error'] / (sketchingRate * normX))
	else:
            newError = np.sqrt(sums['error'] / normX)
	    '''
	    print 'error sum =',sums['error']
	    print 'norm error=',normX
	    print 'new error =',newError
	    '''
        del errorRDD, sums
	errDelta = oldError - newError
	sketchingErrDelta = errDelta
	if sketching > 0:
	    if errDelta > 0.0 and newError < lowestError:
		lowestError = newError
		errorNoChangeCount = 0
	    elif not (sketching in (9,10) and onUpdateWeightLoop):
		errorNoChangeCount = errorNoChangeCount + 1
	    if errDelta > 0.0 and errDelta < minErrDelta:
		errorChangeBelowThreshold = errorChangeBelowThreshold + 1
	    else:
		errorChangeBelowThreshold = 0
	    errDelta = 1 # need to reset this when sketching so it doesn't cause loop to stop
	secondsFromStart = getMS(startSteps, datetime.datetime.now()) / 1000.0

	if sketchingErrDelta > 0.0 and not onUpdateWeightLoop:
	    backupA = A
	    backupB = B
	    backupError = newError
	elif not onUpdateWeightLoop:
	    A = backupA
	    B = backupB
	    newError = backupError

        print 'Iteration',step, ' error =',newError,' sketching rate =',sketchingRate,' delta =',sketchingErrDelta,' secFromStart=',secondsFromStart

	'''
	# calculate full error at each step
	errorRDD = tensorRDD.mapPartitions(saveFactorMatrices)
	sums = errorRDD.reduceByKeyLocally(add)
	newError = np.sqrt(sums['error'] / normX)
	print 'Full Err',step,' curr err=',newError
	'''

	'''
	if newError == 1.0:
	    print 'FULL ERROR'
#	    print '      Sums =',sums
#	    print 'A:\n',A
#	    print 'B:\n',B
	'''
#	if errDelta < minErrDelta or errorNoChangeCount >= maxItersWithoutErrorChange or errorChangeBelowThreshold >= maxItersWithErrorChangeBelowThreshold:
#	    break
#	if secondsFromStart > 300:	# 300 sec, 5 min
	if secondsFromStart > 600:	# 600 sec, 10 min
#	if secondsFromStart > 1200:	# 1200 sec, 20 min
	    break

	# if we don't stop, update sketching rate, if needed
	if sketching == 7 and step > 0:
	    #print 'err delta =', sketchingErrDelta
	    #print 'timestep =', getMS(mabStart, datetime.datetime.now())
	    reward = sketchingErrDelta / (getMS(mabStart, datetime.datetime.now()) / 1000.0)
	    #print 'reward =', reward
	    oldRate = sketchingRate
	    if step < 4:
		runningAve = runningAve + reward
	    elif step == 4:
		runningAve = runningAve / 4.0
		print 'RUNNING AVE',runningAve
		# nadia
		initializeArms(4,runningAve)
	    else:
		sketchingRate = updateMABPosteriorAndSelectNextArm (sketchingRate, reward, step)
	    #print 'Selected new MAB sketching rate: ',sketchingRate
	    if sketchingRate != oldRate:
		sketchingRows, sketchingRows_square_root, sketchingRows_P, sketchingRows_P_cube_root = calculateSketchingValues (sketchingRate)
	elif sketching == 8:
	    sketchingRate = selectRandomRate (step)
	elif sketching in (9,10):
	    reward = sketchingErrDelta / (getMS(mabStart, datetime.datetime.now()) / 1000.0)
	    if not onUpdateWeightLoop:
		sample = np.random.rand()
		print 'MWU epsilon:',mwuEpsilon
		print 'sample:',sample

		if sample < mwuEpsilon:
		    onUpdateWeightLoop = True
		    backupA = A
		    backupB = B
		    backupError = newError
		    # I need some way to know when to select the first arm to update weights
		    sketchingRate = -1.0
	    #print 'reward =', reward
	    oldRate = sketchingRate
	    '''
	    # dalia
	    if newError <= 0.0111:
		sketchingRate = 1.0
	    else:
	    '''
	    sketchingRate = updateMWUWeightAndSelectNextArm (sketchingRate, reward)
	    if sketchingRate != oldRate:
		sketchingRows, sketchingRows_square_root, sketchingRows_P, sketchingRows_P_cube_root = calculateSketchingValues (sketchingRate)

#	elif sketching >= 4 and sketchingRate < finalSketchingRate:
	elif sketching >= 4:
	    oldRate = sketchingRate
	    sketchingRate = updateSketchingRate(sketchingRate, sketchingErrDelta, step)
	    if sketchingRate != oldRate:
		sketchingRows, sketchingRows_square_root, sketchingRows_P, sketchingRows_P_cube_root = calculateSketchingValues (sketchingRate)

	if sketchingErrDelta > 0.0:
	    oldError = newError

    endAll = datetime.datetime.now()

    # calculate the final true error and save the factor matrices
    errorRDD = tensorRDD.mapPartitions(saveFactorMatrices)
    sums = errorRDD.reduceByKeyLocally(add)
    newError = np.sqrt(sums['error'] / normX)

    '''
    print '\n\nFinal Results:\n--------------'
    print 'A:\n--\n',A
    print 'B:\n--\n',B
    '''

    sc.stop ()

    print ''
    print 'Run summary\n-----------'
    print 'Input directory:', inputDir
    if outputDir!='':
	print 'Output directory:', outputDir
    print 'Input tensor dimensions:'
    print '    I =',I,', J =',J,', K =',K
    print '    # entries in tensor:',I*J*K
    print 'Tensor rank:', R
    if regularization==0:
        print 'Regularization: None'
    elif regularization==1:
        print 'Regularization: L2/Tikhonov'
        print '    Regularization parameter:',regulParam
    elif regularization==2:
        print 'Regularization: Proximal'
        print '    Regularization parameter:',regulParam
    if sketching==0:
	print 'Sketching: None'
    elif sketching==1:
	print 'Sketching: CPRAND'
    elif sketching==2:
	print 'Sketching: Random slice selection'
	print '    Sketching rate:',sketchingRate
    elif sketching==3:
	print 'Sketching: Random entry selection'
	print '    Sketching rate:',sketchingRate
    elif sketching==4:
	print 'Sketching: Increase sketching rate x2 every 2 iterations'
	print '    Initial sketching rate:',initialSketchingRate
	print '    Final   sketching rate:',sketchingRate
    elif sketching==5:
	print 'Sketching: Increase sketching rate by buckets iterations'
	print '    Initial sketching rate:',initialSketchingRate
	print '    Final   sketching rate:',sketchingRate
    elif sketching==6:
	print 'Sketching: Increase sketching rate by function'
	print '    Initial sketching rate:',initialSketchingRate
	print '    Final   sketching rate:',sketchingRate
    elif sketching==7:
	print 'Sketching: Multi-Armed Bandit approach'
	print '    First sketching rate:',initialSketchingRate
	print '    Last  sketching rate:',sketchingRate
    elif sketching==8:
	print '    Sketching: Random rate selection'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
    elif sketching==9:
	print '    Sketching: Multiplicative Weight Updates (Label Efficient Forecaster) (random entry selection)'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
	print '    Epsilon:',mwuEpsilon
	print '    Eta    :',mwuEta
    elif sketching==10:
	print '    Sketching: Multiplicative Weight Updates (Label Efficient Forecaster) (row norm weighted sampling)'
	print '    Initial sketching rate:',sketchingRate
	print '    Final   sketching rate:',finalSketchingRate
	print '    Epsilon:',mwuEpsilon
	print '    Eta    :',mwuEta
    elif sketching==11:
	print '    Sketching: Row norm weighted entry selection'
	print '    Sketching rate:',sketchingRate
    print 'Error Calc sketching rate:',errorCalcSketchingRate
    print 'Final error:',newError
    step = step + 1
    print 'Number of iterations:',step
    totalRuntime = getMS(startAll, endAll) / 1000.0
    totalSteptime = getMS(startSteps, endAll) / 1000.0
    mapTime = mapTime / 1000.0
    errorTime = errorTime / 1000.0
    print 'Total runtime (sec):',totalRuntime
    print 'Average runtime (sec):','{0:.6f}'.format(totalRuntime/step)
    print 'Total map time (sec):',mapTime
    print 'Average map time (sec):','{0:.6f}'.format(mapTime/step)
    print 'Total error calc time (sec):',errorTime
    print 'Average error calc time (sec):','{0:.6f}'.format(errorTime/step)
    print 'Total steptime (sec):',totalSteptime
    print 'Average steptime (sec):','{0:.6f}'.format(totalSteptime/step)

    print '\nKSA,',inputDir,',',I,',',J,',',K,',',R,',',regularization,',',regulParam,',',sketching,',',initialSketchingRate,',',sketchingRate,',',step,',',newError,',',totalRuntime,',',(totalRuntime/step),',',totalSteptime,',',(totalSteptime/step),',',mapTime,',',(mapTime/step),',',errorTime,',',(errorTime/step)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Canonical Polyadic Decomposition using Multiplicative Weight Updates.')
    parser.add_argument('-I', '--inputDir', help='Input directory (hdfs or local)', type=str, required=False, default=inputDir)
    parser.add_argument('-O', '--outputDir', help='Output directory (hdfs or local)', type=str, required=False, default=outputDir)
    parser.add_argument('-R', '--r', help='Tensor rank or number of components in decomposition', type=int, required=False, default=R)
    parser.add_argument('-M', '--maxIter', help='Maximum number of iterations (stopping criteria)', type=int, required=False, default=maxIter)
    parser.add_argument('-D', '--minErrDelta', help='Minimum error delta (stopping criteria)', type=float, required=False, default=minErrDelta)
    parser.add_argument('-G', '--regularization', help='Regularization type: 0 = None, 1 = L2/Tikhonov, 2 = Proximal', type=int, required=False, default=regularization)
    parser.add_argument('-L', '--regulParam', help='Regularization parameter (lambda)', type=float, required=False, default=regulParam)
    parser.add_argument('-S', '--sketching', help='Sketching mode: 0 = None, 1 = CPRAND, 2 = Slice sampling by rate, 3 = Entry sampling by rate, 4 = Fixed schedule, 5 = Fixed buckts, 6 = Fixed function, 7 = Multi-Armed Bandit', type=int, required=False, default=sketching)
    parser.add_argument('-K', '--sketchingRate', help='Sketching rate (e.g., 0.1 for 10%)', type=float, required=False, default=0)
    parser.add_argument('-F', '--finalSketchingRate', help='Final sketching rate (e.g., 0.1 for 10%)', type=float, required=False, default=0)
    parser.add_argument('-C', '--errorCalcSketchingRate', help='Error calculation sketching rate (e.g., 0.1 for 10%).  If this is 0 or empty and the sketchingRate is > 0, this value will be set to the sketchingRate.  Set to 1 if you wish to calculate the error across the entire tensor at each step.', type=float, required=False, default=1)
    parser.add_argument('-E', '--numExecutors', help='Number of Spark executors', type=int, required=False, default=numExec)
    parser.add_argument('-eps', '--mwuEpsilon', help='Multiplicative Weight Update epsilon', type=float, required=False, default=mwuEpsilon)
    parser.add_argument('-H', '--mwuEta', help='Multiplicative Weight Update eta', type=float, required=False, default=mwuEta)
    parser.add_argument('-Sd', '--seed', help='Random seed', type=int, required=False, default=0)
    args = parser.parse_args()
#    print (args)
    R = args.r
    inputDir = args.inputDir
    outputDir = args.outputDir
    maxIter = args.maxIter
    minErrDelta = args.minErrDelta
    regularization = args.regularization
    regulParam  = args.regulParam
    sketching = args.sketching
    sketchingRate = args.sketchingRate
    finalSketchingRate = args.finalSketchingRate
    errorCalcSketchingRate = args.errorCalcSketchingRate
    if (errorCalcSketchingRate == 0) and ((finalSketchingRate > 0) or (sketchingRate > 0)):
	if finalSketchingRate > 0:
	    errorCalcSketchingRate = finalSketchingRate
	else:
	    errorCalcSketchingRate = sketchingRate
    numExec = args.numExecutors
    randomSeed = args.seed
    mwuEpsilon = args.mwuEpsilon
    mwuEta = args.mwuEta

    # Make sure the output directory exists and is empty
    if outputDir!='':
	subprocess.call(['hadoop fs -rm -r -skipTrash ' + outputDir], shell=True)
	subprocess.call(['hadoop fs -mkdir ' + outputDir], shell=True)
	subprocess.call(['hadoop fs -chmod 777 ' + outputDir], shell=True)

    parafac_als(inputDir, outputDir, numExec, R, maxIter, minErrDelta, regularization, regulParam, sketching, randomSeed)

