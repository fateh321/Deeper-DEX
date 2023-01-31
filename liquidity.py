import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#function to fetch tick from pmin, pmax and price. Tick at pmin is 0
def fetch_tick(_pmin, _pmax, _price, _bins):
	return math.floor((_price-_pmin)*(_bins-1)/(_pmax-_pmin))

# fetch eth from the tick
def fetch_eth(_pmin, _pmax, liq):
	return liq*(math.sqrt(_pmax) - math.sqrt(_pmin))
# fetch asset from the tick
def fetch_asset(_pmin, _pmax, liq):
	return (liq/math.sqrt(_pmin) - liq/math.sqrt(_pmax))	
#fetch lower price of an interval
def fetch_pLow(_tick, _pmin, _pmax, _bins):
	return _pmin + _tick*(_pmax -_pmin)/_bins

def fetch_pUp(_tick, _pmin, _pmax, _bins):
	return _pmin + (_tick+1)*(_pmax -_pmin)/_bins


#(deprecated)df_temp = pd.read_csv('ALICEUSDT-1m-2023-01-07.csv',names=["Time","Alice","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
# df = df_temp[df_temp.columns[1]]
# df_temp = pd.read_csv('AUDIOUSDT-1m-2023-01-07.csv',names=["Time","Audio","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
# df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
# df_temp = pd.read_csv('ATOMUSDT-1m-2023-01-07.csv',names=["Time","Atom","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
# df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('binance-uniswap-v3/BATETH-1m-2022-12.csv',names=["Time","Bat","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = df_temp[df_temp.columns[1]]
df_temp = pd.read_csv('binance-uniswap-v3/BETAETH-1m-2022-12.csv',names=["Time","Beta","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('binance-uniswap-v3/DARETH-1m-2022-12.csv',names=["Time","Dar","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('binance-uniswap-v3/DENTETH-1m-2022-12.csv',names=["Time","Dent","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('binance-uniswap-v3/GALETH-1m-2022-12.csv',names=["Time","Gal","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('binance-uniswap-v3/HOTETH-1m-2022-12.csv',names=["Time","Hot","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('binance-uniswap-v3/OMGETH-1m-2022-12.csv',names=["Time","OMG","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('binance-uniswap-v3/SLPETH-1m-2022-12.csv',names=["Time","SLP","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)


names = ['BAT','BETA','DAR','DENT','GAL','HOT','OMG','SLP']
# profile the min, max and starting price, and prices of all the assets: should be float
pmin = df.min().to_numpy()
pmax = df.max().to_numpy()
pstart = df.iloc[0].to_numpy()
price = df.to_numpy()

#number of bins (also known as ranges/intervals between ticks) between pmin/r and r*pmax where pmin and pmax are price ranges.

bins = 51
# r is the spread of liquidity.
r = 1.
# 1 day is 1440 mins. We assume same liquidity profile for 1 day, 3 days, 10 days
# iterations = 14400
iterations = 44600

#number of assets. We shall use 3, 5, 10 assets
assets = 5

#total number of liquidity values for share liquidity. We can change this by taking different values for initial available reserves.
Lnum = assets

# collection of available and busy reserves of ETH for each asset
availETHRsrv = 0.1
#busyETHRsrv = []
availAssetRsrv = 0.1*np.ones(assets)
#initial available ETH reserve
availETHRsrvInit = availETHRsrv
#busyAssetRsrv = []
# Virtual Liquidity for each tick: bin x asset. Note that it is L and x*y = L^2
#L = [][]
# Virtual skeleton Liquidity for each tick: bin x asset. Note that it is L and x*y = L^2
sL = np.zeros((bins,assets))
# Virtual individual liquidity
iL = np.zeros((bins,assets))
#initialize skeleton liquidity such that it increases until pstart and then becomes zero
def initial_skeleton_lq():
	for a in range(assets):
		startTck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
		for i in range(startTck+1):
			# sL[i][a] = 1 # uniform liquidity
			sL[i][a] = pow(2, -0.005*abs(i-startTck))

def initial_indv_lq():
	for a in range(assets):
		startTck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
		for i in range(startTck+1, bins):
			# iL[i][a] = 1 # uniform profile
			iL[i][a] = pow(2, -0.005*abs(i-startTck))	
	

initial_skeleton_lq()
initial_indv_lq()

#Total individual liquidity
iLTotal = np.zeros(assets)
#initizilize total individual liquidity
for a in range(assets):
	for i in range(0, bins):
		iLTotal[a] += iL[i][a]

#Total available ETH skeleton reserves (does not count busy reserves)
sETH = np.zeros(assets)
#initialize skeleton ETH
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck):
		sETH[a] += fetch_eth(fetch_pLow(_tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(_tck,pmin[a]/r,r*pmax[a],bins), sL[_tck][a])
	print(sETH[a])
#skeleton eth initially.
sETHInit = np.copy(sETH)		
#Total available asset skeleton reserves (does not count busy reserves)
sAsset = np.zeros(assets)
#initialize skeleton assets
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck+1,bins):
		sAsset[a] += fetch_asset(fetch_pLow(_tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(_tck,pmin[a]/r,r*pmax[a],bins), sL[_tck][a])
	print(sAsset[a])
# Current tick liquidity (L)
currSharedL = 0.001*np.ones(assets)
currIndL = np.zeros(assets)
#initialize current individual liquidity
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	currIndL[a] = iL[tck][a]

# gamma which is the ratio of actual liquidity vs skeleton liquidity when an interval is secured
gamma = np.zeros((bins, assets))

#initial value of current liquidity
currSharedLInit = np.copy(currSharedL)
def initilize_gamma():
	for a in range(assets):
		tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
		gamma[tck][a] = currSharedL[a]/sL[tck][a]
		# print(gamma[tck][a])

initilize_gamma()

# cumulative skeleton liquidity less than the active tick
belowL = np.zeros(assets)
#initialize cumulative skeleton liquidity
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck):
		belowL[a] += sL[_tck][a]
# initial value of belowL
belowLInit = np.copy(belowL)
# cumulative shared liquidity more than the active tick
aboveL = np.zeros(assets)

#initialize cumulative shared liquidity
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck+1,bins):
		aboveL[a] += sL[_tck][a]*gamma[_tck][a]
# initial value of aboveL
aboveLInit = np.copy(aboveL)

#Total actual liquidity for a pair so far: when divided by total iterations give average liquidity
totalL = np.zeros(assets) 
#Liquidity to be ploted against time
L = np.zeros((iterations,assets))
# liquidity boost for the current interval and different values of Lnum
boost = np.ones((assets, Lnum, iterations))


		
#loop through time and update liquidity of each bin
for t in range(1,iterations):
	for a in range(assets):
		_prevtick = fetch_tick(pmin[a]/r,r*pmax[a],price[t-1][a], bins)
		_tick = fetch_tick(pmin[a]/r,r*pmax[a],price[t][a], bins)
		if _tick > _prevtick:
			#for every tick that becomes available, increase ETH reserves from shared pool (gamma); update current liquidity
			for tck in range(_prevtick,_tick):
				# if a==0:
					# print(tck)
				
				#increase available ETH reserve
				_sharedLiq = sL[tck][a]*gamma[tck][a]
				# print(gamma[tck][a])
				availETHRsrv += fetch_eth(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), _sharedLiq)
				#update active liquidity
				currSharedL[a] = gamma[tck+1][a]*sL[tck+1][a]
				currIndL[a] = iL[tck+1][a]
				#(deprecated) currL[a] = sL[tck+1][a]*availAssetRsrv[a]/sAsset[a]
				#(deprecated) decrease available asset reserves
				#(deprecated) availAssetRsrv[a] -= fetch_asset(fetch_pLow(tck+1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck+1,pmin[a]/r,r*pmax[a],bins), currL[a])
				#update sum of skeleton liquidity from below and shared liquidity above the active tick (for future purposes)
				belowL[a] += sL[tck][a]
				aboveL[a] -= currSharedL[a]
				#update sum of skeleton asset reserves below and above the active tick.
				sETH[a] += fetch_eth(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), sL[tck][a])
				#(deprecated) sAsset[a] -= fetch_asset(fetch_pLow(tck+1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck+1,pmin[a]/r,r*pmax[a],bins), sL[tck+1][a])
		elif _tick < _prevtick:
			#for every tick that become unavailable, secure ETH reserves by updating gamma, decrease total ETH reserves; modify liquidity of other assets
			for tck in range(_prevtick,_tick, -1):
				# if a==0:
					# print(tck)
				#(deprecated) increase available Asset reserve
				#(deprecated) availAssetRsrv[a] += fetch_asset(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), currL[a])
				#update active liquidity
				currSharedL[a] = sL[tck-1][a]*availETHRsrv/sETH[a]
				# print(sL[tck-1][a],availETHRsrv,sETH[a],currSharedL[a])
				currIndL[a] = iL[tck-1][a]
				#update gamma
				gamma[tck-1][a] = availETHRsrv/sETH[a]
				# print(gamma[tck-1][a])
				#decrease available ETH reserves
				availETHRsrv -= fetch_eth(fetch_pLow(tck-1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck-1,pmin[a]/r,r*pmax[a],bins), currSharedL[a])
				#update sum of skeleton liquidity below and shared liq above the active tick (for future purposes)
				belowL[a] -= sL[tck-1][a]
				aboveL[a] += sL[tck][a]*gamma[tck][a]
				#update sum of skeleton asset reserves below and above the active tick.
				sETH[a] -= fetch_eth(fetch_pLow(tck-1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck-1,pmin[a]/r,r*pmax[a],bins), sL[tck-1][a])
				#(deprecated) sAsset[a] += fetch_asset(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), sL[tck][a])
		totalL[a] += currSharedL[a]+belowL[a]*availETHRsrv/sETH[a] + aboveL[a] + iLTotal[a]

		# L[t][a] = currSharedL[a]+belowL[a]*availETHRsrv/sETH[a] + aboveL[a] + iLTotal[a]
		L[t][a] = currSharedL[a]+belowL[a]*availETHRsrv/sETH[a] + aboveL[a] + iLTotal[a]
		for n in range(Lnum):
			naiveLiq = availETHRsrvInit*sL[_tick][a]/sETHInit[a]+iL[_tick][a]
			currentLiq = (1+n*(assets-1)/(Lnum-1))*currSharedL[a]+iL[_tick][a]
			# print(_tck)
			# print(naiveLiq,currentLiq, currSharedL[a])
			boost[a][n][t] = currentLiq/naiveLiq
		# L[t][a] = currSharedLInit[a]+belowLInit[a]*availETHRsrvInit/sETHInit[a] + aboveLInit[a]

def plotBoostChart():
	x = np.arange(1,assets+1,1)/assets
	for a in range(assets):
		avg_boost = np.mean(boost[a,:,1:1440], axis=1)
	# plt.plot(boost[0][0], label=names[0])
	# plt.plot(boost[0][Lnum-1], label=names[0])
		print(avg_boost)
		print("eth to get same liq is",(1/assets) + (1-avg_boost[0])*(1-1/assets)/(avg_boost[assets-1]-avg_boost[0]))
		plt.plot(x,avg_boost, label=names[a], marker="^")
	# print(avg_boost)
	plt.xlabel('ETH fraction', fontweight ='bold', fontsize = 18)
	plt.ylabel('Average liquidity boost', fontweight ='bold', fontsize = 18)
	plt.yticks(fontsize=12)
	plt.xticks(fontsize=12)
	plt.legend(fontsize = 14)
	filename = 'Liquidity-boost-wrt-ETH-{}-assets.pdf'.format(assets)
	plt.savefig(filename,bbox_inches='tight')
	plt.show()

# plotBoostChart()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i],y[i]+0.15,float(math.floor(10*y[i]))/10, ha = 'center',fontsize = 18) 

def plotBoostTime():
	boost1day = np.zeros(Lnum)
	boost7day = np.zeros(Lnum)
	boost14day = np.zeros(Lnum)

	for n in range(Lnum):
		boost1day[n] = np.mean(boost[:,n,1:1440])
		boost7day[n] = np.mean(boost[:,n,1:1440*7])
		boost14day[n] = np.mean(boost[:,n,1:1440*14])
	# set width of bar
	barWidth = 0.25
	fig = plt.subplots(figsize =(12, 4))
	# set height of bar in order: each array should be of size assets

	# Set position of bar on X axis
	br1 = np.arange(len(boost1day))
	br2 = [x + barWidth for x in br1]
	br3 = [x + barWidth for x in br2]
	f = plt.figure()
	f.set_figwidth(16)
	f.set_figheight(9)

	addlabels(br1, boost14day)
	addlabels(br2, boost7day)
	addlabels(br3, boost1day)
	# Make the plot
	plt.bar(br1, boost14day, color ='black', width = barWidth,
    	    edgecolor ='grey', label ='T=14 days')
	plt.bar(br2, boost7day, color ='teal', width = barWidth,
    	    edgecolor ='grey', label ='T=7 days')
	plt.bar(br3, boost1day, color ='cyan', width = barWidth,
    	    edgecolor ='grey', label ='T=1 day')
	# Adding Xticks
	plt.xlabel('ETH fraction', fontweight ='bold', fontsize = 24)
	plt.ylabel('Average liquidity boost', fontweight ='bold', fontsize = 24)
	plt.xticks([r + barWidth for r in range(len(boost1day))],
        ['0.2','0.4','0.6','0.8','1.0'])
 
	plt.legend(fontsize = 19)
	plt.xticks(fontsize=24)
	plt.yticks(fontsize=20)
	plt.ylim([0, math.ceil(np.max(boost1day)*1.)+1])
	filename = 'Liquidity-boost-day-comparison-{}-assets.pdf'.format(assets)
	plt.savefig(filename,bbox_inches='tight')
	# plt.plot(boost)
	plt.show()
plotBoostTime()
	

boost1day = np.zeros(assets)
boost10day = np.zeros(assets)
for a in range(assets):
	oldL = currSharedLInit[a]+belowLInit[a]*availETHRsrvInit/sETHInit[a] + aboveLInit[a] +iLTotal[a]
	totalETH = assets*availETHRsrvInit
	naive = oldL/totalETH
	new = np.average(L[1:1441,a])/availETHRsrvInit
	boost1day[a] = new/naive
	new = np.average(L[1:14400,a])/availETHRsrvInit
	boost10day[a] = new/naive
xpoints = np.array([0, iterations-1])

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i],y[i]+0.15,float(math.floor(10*y[i]))/10, ha = 'center',fontsize = 15) 

def plotLiquidity():
	time = 1440
	for a in range(assets):
		plt.plot(L[1:time,a], label=names[a])
	plt.legend(fontsize = 9)
	plt.xlabel('Time (mins)', fontweight ='bold', fontsize = 15)
	plt.ylabel('Virtual liquidity', fontweight ='bold', fontsize = 15)
	plt.xlim([0, time*1.3])
	# f = plt.figure()
	# f.set_figwidth(16)
	# f.set_figheight(5)
	# plt.plot(L[1:x,0], label="names[]")
	# plt.plot(L[1:x,1])
	# plt.plot(L[1:x,2])
	# plt.plot(L[1:x,3])
	# plt.plot(L[1:x,4])
	# plt.plot(L[1:x,5])
	# plt.plot(L[1:x,6])
	# plt.plot(L[1:x,7])
	filename = 'Liquidity-profile-{}-time-5-assets.pdf'.format(time)
	plt.savefig(filename,bbox_inches='tight')
	plt.show()

def plotBoost():
	# set width of bar
	barWidth = 0.25
	fig = plt.subplots(figsize =(12, 4))
	# set height of bar in order: each array should be of size assets

	# Set position of bar on X axis
	br1 = np.arange(len(boost1day))
	br2 = [x + barWidth for x in br1]
	f = plt.figure()
	f.set_figwidth(16)
	f.set_figheight(5)

	addlabels(br1, boost1day)
	addlabels(br2, boost10day)
	# Make the plot
	plt.bar(br1, boost1day, color ='teal', width = barWidth,
    	    edgecolor ='grey', label ='T=1 day')
	plt.bar(br2, boost10day, color ='cyan', width = barWidth,
    	    edgecolor ='grey', label ='T=10 days')
	# Adding Xticks
	plt.xlabel('Token', fontweight ='bold', fontsize = 20)
	plt.ylabel('Liquidity boost', fontweight ='bold', fontsize = 20)
	plt.xticks([r + barWidth for r in range(len(boost1day))],
        names[0:assets])
 
	plt.legend(fontsize = 17)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=15)
	plt.ylim([0, math.ceil(np.max(boost1day)*1.3)+1])
	filename = 'Liquidity-boost-{}-assets.pdf'.format(assets)
	plt.savefig(filename,bbox_inches='tight')
	# plt.plot(boost)
	# plt.show()

# plotBoost()
def std():
	std_dev = np.std(L[1:14400],axis=0)
	print(std_dev)
# std()		
# plotLiquidity()