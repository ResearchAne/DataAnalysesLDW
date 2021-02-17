
# This is code used to create data sets analysed in publications by Ane Dalsnes Storsæter.
# The code shows examples of merging vehicle data with different sample rates, continuous and discrete, and
# the creation of data sets used to perform binary logistic regressions to correlate data from a lane
# departure warning (LDW) system and a mobile retroreflectometer.

import pandas as pd
import numpy as np
#from datetime import datetime
import datetime as dt
#import traces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn as sk  
import pandas as pd  
import xticks
GPSdata= pd.read_csv(r"VehicleData.txt",usecols=[n,n+1,n+2],low_memory=False,names=['T','Longitude','Latitude'])
DiscreteLane= pd.read_csv(r"VehicleData.txt",usecols=[x,x+1],low_memory=False,names=['T2','LaneDetn'])
VehicleSpeed= pd.read_csv(r"VehicleData.txt",usecols=[y,y+1],low_memory=False,names=['T3','VehSpeed'])
AmbLight= pd.read_csv(r"VehicleData.txt",usecols=[z,z+1],low_memory=False,names=['TAmb','AmbLight'])
GPSdata = GPSdata.dropna(how='all') 
DiscreteLane = DiscreteLane.dropna(how='all')
VehicleSpeed = VehicleSpeed.dropna(how='all')
AmbLight = AmbLight.dropna(how='all')

#Set date to today's date but leave time stamp:
dateUsed = '2019-08-30 21:30:44'
dateCorr = '2019-08-30 21:28:00'

#Lane detection is the sampling rate to match due to its categorical nature. Resampling lane detection will create
# falsely continuous values.
#Therefore, we want to downsample vehicle speed and average the values, upsample lat and long with interpolation, 
# and upsample retromeasurements with interpolation.

#Converting time to datetime
VehicleSpeed.index = pd.to_datetime(VehicleSpeed['T3'], unit='s',origin=pd.Timestamp(dateUsed))
GPSdata.index = pd.to_datetime(GPSdata['T'], unit='s',origin=pd.Timestamp(dateUsed))
DiscreteLane.index = pd.to_datetime(DiscreteLane['T2'], unit='s',origin=pd.Timestamp(dateUsed))
AmbLight.index = pd.to_datetime(AmbLight['TAmb'], unit='s',origin=pd.Timestamp(dateUsed))

#Clean up columns.
GPSdata['Time']= GPSdata['T']
VehicleSpeed['Time']= VehicleSpeed['T3']
del GPSdata['T']
del VehicleSpeed['T3']

#Merging GPS and Vehicle speed with interpolation.
GPSVehSpeed = pd.merge_ordered(GPSdata, VehicleSpeed, left_on='T', right_on='T3', how='outer',fill_method='ffill')
GPSVehSpeed.head(15)

#Clean up columns.
DiscreteLane['Time']= DiscreteLane['T2']
del DiscreteLane['T2']

# Making index datetime format.
GPSVehSpeed.index = pd.to_datetime(GPSVehSpeed['Time_y'], unit='s',origin=pd.Timestamp(dateUsed))

#Clean up columns.
GPSVehSpeed['Time'] = GPSVehSpeed['Time_y']
del GPSVehSpeed['Time_y']

# Merge gps data, vehicle speed and discrete lane data with interpolation.
GPSSpeedLane = pd.merge_ordered(GPSVehSpeed, DiscreteLane, left_on='Time', right_on='Time', how='outer',
                                fill_method='ffill')

#Clean up columns.

AmbLight['Time']= AmbLight['TAmb']
del AmbLight['TAmb']

# Merge gps, speed, lane det with ambient light.
GPSSpeedLaneAmb = pd.merge_ordered(GPSSpeedLane, AmbLight, left_on='Time', right_on='Time', how='outer',
                                   fill_method='ffill')
GPSSpeedLaneAmb.head(15)

#Plot for sanity check:
GPSSpeedLaneAmb.plot(x='Time', y= 'AmbLight')

#Done merging vehicle data at different sample rates, now to add laser data (retroreflectometer).
# Reading files:
df = pd.read_csv(r"LaserDataPart1.csv",sep=';',skiprows=26,usecols = ['Time', 'Latitude','Longitude',
                                                                      'Valid_Scans_Left','Retro_Left_Average',
                                                                      'Retro_Left_Contrast','Odometer','Vehicle Speed'])
df2 = pd.read_csv(r"LaserDataPart2.csv",sep=';',skiprows=26,usecols = ['Time', 'Latitude','Longitude',
                                                                       'Valid_Scans_Left','Retro_Left_Average',
                                                                    'Retro_Left_Contrast','Odometer','Vehicle Speed'])

# Making sure vehicle speed is in the same units for vehicle and retroreflectometer:
df['Vehicle Speed']=0.277777778*df['Vehicle Speed']
df2['Vehicle Speed']=0.277777778*df2['Vehicle Speed']

# Concatenate data sets to get the data set that matches the vehicle data in time:
frames = [df, df2]
laserFullBothWays = pd.concat(frames,ignore_index=True)

# Changing time columns to datetime format for both vehicle and retroreflectometer data:
laserFullBothWays['TimePlot'] = pd.to_datetime(laserFullBothWays['Time'])
GPSSpeedLaneAmb['TimeMerge'] = pd.to_datetime(GPSSpeedLaneAmb['Time'], unit='s',origin=pd.Timestamp(dateCorr))


# Sanity check:
laserFullBothWays.dtypes

GPSSpeedLaneAmb.dtypes

# Merge Vehicle data with retroreflectometer data:
VehicleLaser = pd.merge_ordered(GPSSpeedLaneAmb, laserFullBothWays, left_on='TimeMerge', right_on='TimePlot',
                               fill_method='ffill', how='outer')
VehicleLaser.head(5)


# Plotting to check that data is overlapping and correct:
ax3 = VehicleLaser.plot(x='Time_y', y='VehSpeed',linewidth=3.3) 
ax4 = VehicleLaser.plot(x='Time_y', y='Vehicle Speed',color='g', ax=ax3) 

print(ax3 == ax4)

# Position data (lat,lon) is not always accurate. Performing additional sanity check that the merge is correct
# by correlation vehicle speed measure in vehicle and by retroreflectometer:
VehicleLaser['VehSpeed'].max()
VehicleLaser[['VehSpeed']].idxmax() 
print(VehicleLaser[VehicleLaser.VehSpeed == VehicleLaser.VehSpeed.max()]) 
print(VehicleLaser[VehicleLaser['Vehicle Speed'] == VehicleLaser['Vehicle Speed'].max()]) 
VehicleLaser.shape

# Creating sections of data needed for analyses:
VehicleLaserFormer = VehicleLaser[286940:1200000]
VehicleLaserLatter = VehicleLaser[1200001:2787897]

# Plots to check:
ax321 = VehicleLaserFormer.plot(x='Time_y', y='VehSpeed',linewidth=3.3) 
ax432 = VehicleLaserFormer.plot(x='Time_y', y='Vehicle Speed',color='g', ax=ax321) 
print(ax321 == ax432)

# More plots to check:
ax33 = VehicleLaserLatter.plot(x='Time_y', y='VehSpeed',linewidth=3.3) 
ax44 = VehicleLaserLatter.plot(x='Time_y', y='Vehicle Speed',color='g', ax=ax33) 
print(ax33 == ax44)

# Even more plots to check:
ax01 = VehicleLaserFormer.plot(x='Time_y', y='Latitude_x',linewidth=3.3) 
ax02 = VehicleLaserFormer.plot(x='Time_y', y='Latitude_y',color='r', ax=ax01) 
ax01.set_ylim(59.3,60)
print(ax01 == ax02)

# Some more plots:
ax011 = VehicleLaserFormer.plot(x='Time_y', y='Longitude_x',linewidth=4) 
ax022 = VehicleLaserFormer.plot(x='Time_y', y='Longitude_y',color='r', ax=ax011) 
ax011.set_ylim(10.6,11)
print(ax011 == ax022)

# Some plots:
ax1 = VehicleLaserLatter.plot(x='Time_y', y='Latitude_x',linewidth=3.3) 
ax2 = VehicleLaserLatter.plot(x='Time_y', y='Latitude_y',color='g', ax=ax1) 
ax1.set_ylim(59.3,60)
print(ax1 == ax2)

# Plots:
ax11 = VehicleLaserLatter.plot(x='Time_y', y='Longitude_x',linewidth=3.3) 
ax22 = VehicleLaserLatter.plot(x='Time_y', y='Longitude_y',color='g', ax=ax11) 
ax11.set_ylim(10.6,11)
print(ax11 == ax22)

# Look at discrete lane detection in a histogram:
VehicleLaserLatter.hist(column='LaneDetn')

# Change discrete values to binary values:
laneBinary = []
for value in VehicleLaserLatter["LaneDetn"]: 
    if value == 0: 
        laneBinary.append("0") 
    elif value == 1:
        laneBinary.append("0") 
    elif value == 2:
        laneBinary.append("1") 
    elif value == 3:
         laneBinary.append("1") 
    else: 
        laneBinary.append("0") 

VehicleLaserLatter["laneBinary"] = laneBinary
VehicleLaserLatter["laneBinary"] = VehicleLaserLatter["laneBinary"].astype(int)

#New plot of LaneDet values to validate conversion to binary set
VehicleLaserLatter.hist(column='laneBinary')

# Change discrete values to binary values:
laneBinary2 = []
for value in VehicleLaserFormer["LaneDetn"]: 
    if value == 0: 
        laneBinary2.append("0") 
    elif value == 1:
        laneBinary2.append("0") 
    elif value == 2:
        laneBinary2.append("1") 
    elif value == 3:
         laneBinary2.append("1") 
    else: 
        laneBinary2.append("0") 

VehicleLaserFormer["laneBinary"] = laneBinary2
VehicleLaserFormer["laneBinary"] = VehicleLaserFormer["laneBinary"].astype(int)

#New plot of LaneDet values to validate conversion to binary set
VehicleLaserFormer.hist(column='laneBinary')

# Plot:
ax17 = VehicleLaserFormer.plot(x='Time_y', y='Latitude_x',linewidth=3.3) 
ax27 = VehicleLaserFormer.plot(x='Time_y', y='Latitude_y',color='g', ax=ax17)
print(ax17 == ax27)

# Create files for statistical analyses:
# County road night:
VehicleLaserLatter.to_csv(r'CountyRoadNight.csv')
# Freeway night:
VehicleLaserFormer.to_csv(r'FreewayNight.csv')


