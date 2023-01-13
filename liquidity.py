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


df_temp = pd.read_csv('ALICEUSDT-1m-2023-01-07.csv',names=["Time","Alice","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = df_temp[df_temp.columns[1]]
df_temp = pd.read_csv('AUDIOUSDT-1m-2023-01-07.csv',names=["Time","Audio","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('ATOMUSDT-1m-2023-01-07.csv',names=["Time","Atom","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
# profile the min, max and starting price, and prices of all the assets: should be float
pmin = df.min().to_numpy()
pmax = df.max().to_numpy()
pstart = df.iloc[0].to_numpy()
price = df.to_numpy()

#number of bins (also known as ranges/intervals between ticks) between pmin/r and r*pmax where pmin and pmax are price ranges.

bins = 51
# r is the spread of liquidity.
r = 1.5
# 1 day is 1440 mins. We assume same liquidity profile for 1 day, 3 days, 10 days
iterations = 1400

#number of assets. We shall use 3, 5, 10 assets
assets = 3

# collection of available and busy reserves of ETH for each asset
availETHRsrv = 0.1
#busyETHRsrv = []
availAssetRsrv = 0.1*np.ones(assets)
#busyAssetRsrv = []
# Virtual Liquidity for each tick: bin x asset. Note that it is L and x*y = L^2
#L = [][]
# Virtual skeleton Liquidity for each tick: bin x asset. Note that it is L and x*y = L^2
sL = np.zeros((bins,assets))

#initialize skeleton liquidity
for i in range(bins):
	for j in range(assets):
		sL[i][j] = pow(2, -0.05*abs(i-(bins+1)/2))

#Total available ETH skeleton reserves (does not count busy reserves)
sETH = np.zeros(assets)
#initialize skeleton ETH
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck):
		sETH[a] += fetch_eth(fetch_pLow(_tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(_tck,pmin[a]/r,r*pmax[a],bins), sL[_tck][a])
	print(sETH[a])	
#Total available asset skeleton reserves (does not count busy reserves)
sAsset = np.zeros(assets)
#initialize skeleton assets
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck+1,bins):
		sAsset[a] += fetch_asset(fetch_pLow(_tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(_tck,pmin[a]/r,r*pmax[a],bins), sL[_tck][a])
	print(sAsset[a])
# Current tick liquidity (L)
currL = 0.1*np.ones(assets)
# cumulative skeleton liquidity less than the active tick
belowL = np.zeros(assets)
#initialize cumulative skeleton liquidity
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck):
		belowL[a] += sL[_tck][a]

# cumulative skeleton liquidity more than the active tick
aboveL = np.zeros(assets)
#initialize cumulative skeleton liquidity
for a in range(assets):
	tck = fetch_tick(pmin[a]/r,r*pmax[a],price[0][a], bins)
	for _tck in range(tck+1,bins):
		aboveL[a] += sL[_tck][a]

#Total actual liquidity for a pair so far: when divided by total iterations give average liquidity
totalL = np.zeros(assets) 
#Liquidity to be ploted against time
L = np.zeros((iterations,assets))


		
#loop through time and update liquidity of each bin
for t in range(1,iterations):
	for a in range(assets):
		_prevtick = fetch_tick(pmin[a]/r,r*pmax[a],price[t-1][a], bins)
		_tick = fetch_tick(pmin[a]/r,r*pmax[a],price[t][a], bins)
		if _tick > _prevtick:
			#for every tick that become available, increase available reserves; modify liquidity of other assets
			for tck in range(_prevtick,_tick):
				# if a==0:
					# print(tck)
				
				#increase available ETH reserve
				availETHRsrv += fetch_eth(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), currL[a])
				#update active liquidity
				currL[a] = sL[tck+1][a]*availAssetRsrv[a]/sAsset[a]
				#decrease available asset reserves
				availAssetRsrv[a] -= fetch_asset(fetch_pLow(tck+1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck+1,pmin[a]/r,r*pmax[a],bins), currL[a])
				#update sum of skeleton liquidity below and above the active tick (for future purposes)
				belowL[a] += sL[tck][a]
				aboveL[a] -= sL[tck+1][a]
				#update sum of skeleton asset reserves below and above the active tick.
				sETH[a] += fetch_eth(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), sL[tck][a])
				sAsset[a] -= fetch_asset(fetch_pLow(tck+1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck+1,pmin[a]/r,r*pmax[a],bins), sL[tck+1][a])
		elif _tick < _prevtick:
			#for every tick that become unavailable, decrease available reserves; modify liquidity of other assets
			for tck in range(_prevtick,_tick, -1):
				# if a==0:
					# print(tck)
				#increase available Asset reserve
				availAssetRsrv[a] += fetch_asset(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), currL[a])
				#update active liquidity
				currL[a] = sL[tck-1][a]*availETHRsrv/sETH[a]
				#decrease available ETH reserves
				availETHRsrv -= fetch_eth(fetch_pLow(tck-1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck-1,pmin[a]/r,r*pmax[a],bins), currL[a])
				#update sum of skeleton liquidity below and above the active tick (for future purposes)
				belowL[a] -= sL[tck-1][a]
				aboveL[a] += sL[tck][a]
				#update sum of skeleton asset reserves below and above the active tick.
				sETH[a] -= fetch_eth(fetch_pLow(tck-1,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck-1,pmin[a]/r,r*pmax[a],bins), sL[tck-1][a])
				sAsset[a] += fetch_asset(fetch_pLow(tck,pmin[a]/r,r*pmax[a],bins), fetch_pUp(tck,pmin[a]/r,r*pmax[a],bins), sL[tck][a])
		totalL[a] += currL[a]+belowL[a]*availETHRsrv/sETH[a] + aboveL[a]*availAssetRsrv[a]/sAsset[a]
		L[t][a] = currL[a]+belowL[a]*availETHRsrv/sETH[a] + aboveL[a]*availAssetRsrv[a]/sAsset[a]
		# L[t][a] = availETHRsrv

xpoints = np.array([0, iterations-1])
# plt.plot(np.log(L[:,1]))
plt.plot(L[:,0])
plt.plot(L[:,1])
plt.plot(L[:,2])
plt.show()