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
#matrix containing log of drop of the mean of an asset from the initial value
drop = np.zeros(8)
#for an asset pair (x,y) boost_self is the boost for x where drop matrix also contains x
boost_self = np.zeros([7,8])
#for an asset pair (x,y) boost_other is the boost for y where drop matric contains x
boost_other = np.zeros([7,8])

df_np = df.to_numpy()
interval = 1440*14
for lf in range(8):
	drop[lf] = np.log(df_np[0,lf]/np.mean(df_np[1:interval,lf]))
	_index = 0
	for rt in range(8):
		_df = df[df.columns[lf]]
		_df = pd.concat([_df, df[df.columns[rt]]], axis=1)
		# profile the min, max and starting price, and prices of all the assets: should be float
		pmin = _df.min().to_numpy()
		pmax = _df.max().to_numpy()
		pstart = _df.iloc[0].to_numpy()
		price = _df.to_numpy()

		#number of bins (also known as ranges/intervals between ticks) between pmin/r and r*pmax where pmin and pmax are price ranges.

		bins = 51
		# r is the spread of liquidity.
		r = 1.
		# 1 day is 1440 mins. We assume same liquidity profile for 1 day, 3 days, 10 days
		# iterations = 14400
		iterations = 44600

		#number of assets. We shall use 3, 5, 10 assets
		assets = 2

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
			#uncomment below for safety check
			# print(sETH[a])
		#skeleton eth initially.
		sETHInit = np.copy(sETH)		
		#Total available asset skeleton reserves (does not count busy reserves)
		sAsset = np.zeros(assets)
		#initialize skeleton assets
		for a in range(assets):
			tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
			for _tck in range(tck+1,bins):
				sAsset[a] += fetch_asset(fetch_pLow(_tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(_tck,pmin[a]/r,r*pmax[a],bins), sL[_tck][a])
			#uncomment below for safety check
			# print(sAsset[a])
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
		
		if rt!=lf:
			boost_self[_index,lf] = np.mean(boost[0,1,1:interval])
			boost_other[_index,lf] = np.mean(boost[1,1,1:interval])
			_index+=1

		# drop[_index] = np.log(price[0,0]/np.mean(price[1:interval,0]))
		# drop[_index+1] = np.log(price[0,1]/np.mean(price[1:interval,1]))
		# boost1[_index] = np.mean(boost[0,1,1:interval])
		# boost1[_index+1] = np.mean(boost[1,1,1:interval])	
		# corr[index] = df[df.columns[lf]].iloc[1:1440*25:1].corr(df[df.columns[rt]].iloc[1:1440*25:1])
		# mean[index] = np.mean(boost[:,1,1:1440*25])
		# error[index] = abs(np.mean(boost[0,1,1:1440*25]) -np.mean(boost[1,1,1:1440*25]))/2
		# index+=1
		# _index+=2
		# print(lf,rt,np.mean(boost[0,1,1:1440*25]),np.mean(boost[1,1,1:1440*25]))

# plt.errorbar(corr, mean, error, linestyle='None', marker='^')
# plt.show()
#scatter plot
# plt.scatter(drop,boost1)
# plt.show()
#mean error plot
mean_vec = np.mean(boost_self,axis=0)
# error_vec = (np.amax(boost_self,axis=0)-np.amin(boost_self,axis=0))/2
error_vec = np.std(boost_self, axis=0)
# plt.bar(drop, np.amax(boost_other,axis=0), bottom=np.amin(boost_other,axis=0))
# plt.show()
fig, ax = plt.subplots(figsize=(10, 7))
for asst in range(8):
	# ax.errorbar(drop[asst], mean_vec[asst], yerr=error_vec[asst], fmt = 'o', label = names[asst])
	ax.errorbar(drop[asst], mean_vec[asst], yerr=error_vec[asst], fmt = 'o', markersize=4, capsize=5,capthick=1.5,color="black",markeredgecolor='black',markerfacecolor='None',)
for col in range(8):
	ax.scatter(drop[col]*np.ones(7),boost_self[:,col], marker="x",s=80, label = names[col], linewidths=1.5)
	print(col,drop[col],np.mean(boost_self[:,col]),boost_self[:,col])
# ax.errorbar(drop, mean_vec, yerr=error_vec, fmt = 'o', color='blue',ecolor='lightblue', elinewidth=3, capsize=1)
ax.legend(loc='best',fontsize = 15)
plt.xlabel('Logarithm of price drop', fontweight ='bold', fontsize = 20)
plt.ylabel('Boost for the same asset', fontweight ='bold', fontsize = 20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

filename = 'Liquidity-boost-for-the-same-asset-vs-drop.pdf'
# plt.savefig(filename,bbox_inches='tight')
plt.show()
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i],y[i]+0.15,float(math.floor(10*y[i]))/10, ha = 'center',fontsize = 18) 

def plotBoostRange():
	boost_max = np.amax(boost_other,axis=0)
	boost_min = np.amin(boost_other,axis=0)
	# set width of bar
	barWidth = 0.25
	fig = plt.subplots(figsize =(12, 4))
	# set height of bar in order: each array should be of size assets

	# Set position of bar on X axis
	br = drop
	f = plt.figure()
	f.set_figwidth(16)
	f.set_figheight(9)

	addlabels(drop, boost_max)
	addlabels(drop, boost_min)

	# Make the plot
	plt.bar(drop, boost_max, color ='black', width = barWidth,
    	    edgecolor ='grey', label ='T=14 days', bottom = boost_min)

	# Adding Xticks
	plt.xlabel('Asset', fontweight ='bold', fontsize = 24)
	plt.ylabel('Average liquidity boost for the other', fontweight ='bold', fontsize = 24)
	plt.xticks([r + barWidth for r in range(len(boost_max))],
        ['BAT','BETA','DAR','DENT','GAL','HOT','OMG','SLP'])
 
	plt.legend(fontsize = 19)
	plt.xticks(fontsize=24)
	plt.yticks(fontsize=20)
	plt.ylim([0, math.ceil(np.max(boost_max)*1.)+10])
	# filename = 'Liquidity-boost-day-comparison-{}-assets.pdf'.format(assets)
	# plt.savefig(filename,bbox_inches='tight')
	# plt.plot(boost)
	plt.show()
# plotBoostRange()
# plt.errorbar(drop, mean_vec, error_vec, linestyle='None', marker='^')
# plt.show()
