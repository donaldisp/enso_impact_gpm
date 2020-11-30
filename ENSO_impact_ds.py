# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:03:02 2020

@author: donaldi permana
GAW Palu BMKG
donaldi.permana@bmkg.go.id
"""
# from matplotlib.cbook import dedent
# from mpl_toolkits.basemap import Basemap, maskoceans
import xarray as xr
# import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as pl
import datetime as dt
import pandas as pd
import scipy
from scipy import signal 
import math
import shapely.geometry
# from shapely.geometry import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

shpkabkotfilename = 'C:\\Users\\ASUS\\Desktop\\ENSO_PRECIPITATION\\shp\\Indo_Kab_Kot1'
shpprovfilename = 'C:\\Users\\ASUS\\Desktop\\ENSO_PRECIPITATION\\shp\\INDONESIA_PROP1'
# shpoceanfilename = 'G:\\DATA\\BMKG\\2018_BMKG\\12\\backup\\pyscript\\shp\\ne_10m_ocean'
# nclandseamaskfile = 'G:\\DATA\\BMKG\\2018_BMKG\\12\\backup\\pyscript\\shp\\lsmask.oisst.v2.nc'
# bmkglogo = 'G:\\DATA\\BMKG\\2018_BMKG\\12\\backup\\pyscript\\BMKG.png'

dirin = 'C:\\Users\\ASUS\\Desktop\\ENSO_PRECIPITATION\\IDN\\'
dirout = 'C:\\Users\\ASUS\\Desktop\\ENSO_PRECIPITATION\\'

oniidx = pd.read_csv(dirout+'onidata.csv')  # 1950 - 2020
nino34idx = pd.read_csv(dirout+'nino34data.csv') # 1950 - 2020 

oni = oniidx.values.reshape(71*12)
nino34 = nino34idx.values.reshape(71*12)

# fig = pl.figure()
# pl.plot(oni,'b')
# pl.plot(np.nanmean(oni)+signal.detrend(oni[:-4]),'r')

# fig = pl.figure()
# pl.plot(nino34,'b')
# pl.plot(np.nanmean(nino34)+signal.detrend(nino34[:-4]),'r')

oni = np.nanmean(oni)+signal.detrend(oni[:-4]) #detrending for removing global warming signal
nino34 = np.nanmean(nino34)+signal.detrend(nino34[:-4]) #detrending for removing global warming signal

oni_2001_2019 = oni[612:612+228]
nino34_2001_2019 = nino34[612:612+228]

year = '*' # 2001 - 2019
filename = dirin + 'GPM_' + year + '.nc'

lanina_filename = dirin + '\\LaNina\\GPM_' + year + '.nc'
elnino_filename = dirin + '\\ElNino\\GPM_' + year + '.nc'

# ds = xr.open_dataset(filename) # single file
ds = xr.open_mfdataset(filename) # multiple files
ds_lanina = xr.open_mfdataset(lanina_filename) # multiple files
ds_elnino = xr.open_mfdataset(elnino_filename) # multiple files

# ds_landseamask = xr.open_dataset(nclandseamaskfile) # read land sea mask
# ttime, lats_lsmask, lons_lsmask = ds_landseamask.indexes.values()
# lsmask = ds_landseamask.lsmask.values
# lsmask = np.flipud(lsmask[0,:,:])

# var = 'precipitation'

monthly_sum = ds.resample(time='MS').sum()
monthly_ave = monthly_sum.groupby('time.month').mean('time')
monthly_anom = monthly_sum.groupby('time.month') - monthly_ave # anomaly by removing seasonality
month, lons, lats  = monthly_anom.indexes.values()
# convert Dataset to Array
month_sum = monthly_sum.precipitation.values
month_ave = monthly_ave.precipitation.values
month_anom = monthly_anom.precipitation.values

# fig=pl.figure()
# pl.plot(month_sum[:,0,0])
# pl.plot(month_anom[:,0,0],'k')

# La Nina
monthly_sum_lanina = ds_lanina.resample(time='MS').sum()
monthly_ave_lanina = monthly_sum_lanina.groupby('time.month').mean('time')
monthly_anom_lanina = monthly_ave_lanina - monthly_ave
monthly_anom_lanina_percent = 100*(monthly_anom_lanina/monthly_ave)
# convert Dataset to Array
month_anom_lanina = monthly_anom_lanina.precipitation.values
month_anom_lanina_percent = monthly_anom_lanina_percent.precipitation.values

# El Nino
monthly_sum_elnino = ds_elnino.resample(time='MS').sum()
monthly_ave_elnino = monthly_sum_elnino.groupby('time.month').mean('time')
monthly_anom_elnino = monthly_ave_elnino - monthly_ave
monthly_anom_elnino_percent = 100*(monthly_anom_elnino/monthly_ave)
# convert Dataset to Array
month_anom_elnino = monthly_anom_elnino.precipitation.values
month_anom_elnino_percent = monthly_anom_elnino_percent.precipitation.values

# Prep for Corr Coef and p-values
corr_oni_all = np.zeros((len(lons), len(lats)))
pval_oni_all = np.zeros((len(lons), len(lats)))
corr_nino34_all = np.zeros((len(lons), len(lats)))
pval_nino34_all = np.zeros((len(lons), len(lats)))

corr_oni_season = np.zeros((4, len(lons), len(lats)))
pval_oni_season = np.zeros((4, len(lons), len(lats)))
corr_nino34_season = np.zeros((4, len(lons), len(lats)))
pval_nino34_season = np.zeros((4, len(lons), len(lats)))

corr_oni_month = np.zeros((12, len(lons), len(lats)))
pval_oni_month = np.zeros((12, len(lons), len(lats)))
corr_nino34_month = np.zeros((12, len(lons), len(lats)))
pval_nino34_month = np.zeros((12, len(lons), len(lats)))

# Calculate Corr Coef and p-values
for x in range(len(lons)):
    for y in range(len(lats)):
        prec_anom = month_anom[:,x,y]
        n = len(oni_2001_2019)
        corr_oni_all[x,y] = scipy.stats.pearsonr(prec_anom,oni_2001_2019)[0]
        pval_oni_all[x,y] = scipy.stats.pearsonr(prec_anom,oni_2001_2019)[1]
        corr_nino34_all[x,y] = scipy.stats.pearsonr(prec_anom,nino34_2001_2019)[0]
        pval_nino34_all[x,y] = scipy.stats.pearsonr(prec_anom,nino34_2001_2019)[1]

        for mon in range(12):
            idx = np.arange(mon,n,12)
            corr_oni_month[mon,x,y] = scipy.stats.pearsonr(prec_anom[idx],oni_2001_2019[idx])[0]
            pval_oni_month[mon,x,y] = scipy.stats.pearsonr(prec_anom[idx],oni_2001_2019[idx])[1]
            corr_nino34_month[mon,x,y] = scipy.stats.pearsonr(prec_anom[idx],nino34_2001_2019[idx])[0]
            pval_nino34_month[mon,x,y] = scipy.stats.pearsonr(prec_anom[idx],nino34_2001_2019[idx])[1]

        for season in range(4):
            if season == 0: #DJF
                idx = sorted(np.append(np.append(np.arange(0,n,12),np.arange(1,n,12)),np.arange(11,n,12)))
            if season == 1: #MAM
                idx = sorted(np.append(np.append(np.arange(2,n,12),np.arange(3,n,12)),np.arange(4,n,12)))
            if season == 2: #JJA
                idx = sorted(np.append(np.append(np.arange(5,n,12),np.arange(6,n,12)),np.arange(7,n,12)))
            if season == 3: #SON
                idx = sorted(np.append(np.append(np.arange(8,n,12),np.arange(9,n,12)),np.arange(10,n,12)))
            corr_oni_season[season,x,y] = scipy.stats.pearsonr(prec_anom[idx],oni_2001_2019[idx])[0]
            pval_oni_season[season,x,y] = scipy.stats.pearsonr(prec_anom[idx],oni_2001_2019[idx])[1]
            corr_nino34_season[season,x,y] = scipy.stats.pearsonr(prec_anom[idx],nino34_2001_2019[idx])[0]
            pval_nino34_season[season,x,y] = scipy.stats.pearsonr(prec_anom[idx],nino34_2001_2019[idx])[1]
            
minlon = np.min(lons)
maxlon = np.max(lons)
minlat = np.min(lats)
maxlat = np.max(lats)

# minlon = 119.
# maxlon = 124.5
# minlat = -3.5
# maxlat = 2

domain = 'Indo'
# domain = 'Sulteng'

lsmask = True
# lsmask = False

# kab/kot 
# kab = True
kab = False

# plotting - monthly ave (climatology)
for mon in range(12):
    # fig = pl.figure(figsize=[12,5]) 
    # ax = fig.add_subplot(111, projection=ccrs.Mercator())

    # monthly_ave[var][i].transpose().plot.pcolormesh(
    #     ax=ax, vmin=0, vmax=1500, cmap='Spectral_r',
    #     add_colorbar=True, extend='both')

    # ax.coastlines(color='b', linewidth=2.)
    
    monthstr = dt.date(1900, mon+1, 1).strftime('%B')

    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
    
    # m.etopo()    
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(0,801,50)
    month_ave_tr = np.transpose(month_ave[mon])
    if lsmask == True:
        month_ave_tr = maskoceans(lon,lat,month_ave_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, month_ave_tr, clevs, extend = 'both', cmap=pl.cm.viridis.reversed()) 
    # Add Grid Lines
    m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    m.drawcountries()
    # m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='white', lakes=True)
    # m.readshapefile(shpoceanfilename, 'OCEAN')
    # patches = []
    # for info, shape in zip(m.OCEAN_info, m.OCEAN):
    #     patches.append( Polygon(np.array(shape), True) )
    # ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2))
    if kab == True: 
        m.readshapefile(shpkabkotfilename, 'KABKOT')
        patches = []
        for info, shape in zip(m.KABKOT_info, m.KABKOT):
              max = np.amax(shape, axis=0)
              min = np.amin(shape, axis=0)
              diff = max-min
              londistance = 2*diff[0]
              pol = shapely.geometry.Polygon(shape)
              center = pol.centroid.coords[0]
              centerdiff = center-min
              x1 = center[0]-0.28*centerdiff[0]
              y1 = center[1] 
              #x1 = min[0]+0.4*diff[0] #lon
              #y1 = min[1]+0.2*diff[0] #lat
              #mid = min+diff
              #print mid
              if x1>minlon+0.2 and x1<maxlon-0.2 and y1>minlat+0.2 and y1<maxlat-0.2 and londistance>0.5:
                  if info['PROVINSI'] == 'SULAWESI TENGAH' or info['PROVINSI'] == 'Sulawesi Tengah':
                      x, y = m(x1, y1)
                      #print y1, x1, info['KABKOT']
                      strtext = info['KABKOT']#.replace('JAKARTA PUSAT','JAKPUS').replace('JAKARTA SELATAN','JAKSEL').replace('JAKARTA UTARA','JAKUT').replace('JAKARTA TIMUR','JAKTIM').replace('JAKARTA BARAT','JAKBAR').replace('TANGERANG SELATAN','TANGSEL').replace('TANGERANG','TANG')
                      pl.text(x, y, strtext.replace(' ','\n'),fontsize=8)
                  # else:                      
                      # patches.append( Polygon(np.array(shape), True) )
        # ax = pl.gca()
        # ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2)) 
        m.readshapefile(shpprovfilename, 'PROV')
        patches = []
        for info, shape in zip(m.PROV_info, m.PROV):
            if info['Propinsi'] != 'SULAWESI TENGAH':
                patches.append( Polygon(np.array(shape), True) )
        ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2))
        
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Precipitation mm')
    pl.title('Monthly Mean Precipitation - GPM_IMERGV06 (2001-2019) - ' + monthstr )
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show() 
    fig.savefig(dirout + domain + '-GPM_monthly_mean_' + str(mon+1) + '.png', format='png', dpi=300, bbox_inches='tight')

# plotting - monthly anomaly mm - La Nina
for mon in range(12):
    monthstr = dt.date(1900, mon+1, 1).strftime('%B')

    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-100,101,10)
    month_anom_lanina_tr = np.transpose(month_anom_lanina[mon])
    if lsmask == True:
        month_anom_lanina_tr = maskoceans(lon,lat,month_anom_lanina_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, month_anom_lanina_tr, clevs, extend = 'both', cmap=pl.cm.RdBu) 
    # Add Grid Lines
    m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    m.drawcountries()
    if kab == True: 
        m.readshapefile(shpkabkotfilename, 'KABKOT')
        patches = []
        for info, shape in zip(m.KABKOT_info, m.KABKOT):
              max = np.amax(shape, axis=0)
              min = np.amin(shape, axis=0)
              diff = max-min
              londistance = 2*diff[0]
              pol = shapely.geometry.Polygon(shape)
              center = pol.centroid.coords[0]
              centerdiff = center-min
              x1 = center[0]-0.28*centerdiff[0]
              y1 = center[1] 
              #x1 = min[0]+0.4*diff[0] #lon
              #y1 = min[1]+0.2*diff[0] #lat
              #mid = min+diff
              #print mid
              if x1>minlon+0.2 and x1<maxlon-0.2 and y1>minlat+0.2 and y1<maxlat-0.2 and londistance>0.5:
                  if info['PROVINSI'] == 'SULAWESI TENGAH' or info['PROVINSI'] == 'Sulawesi Tengah':
                      x, y = m(x1, y1)
                      #print y1, x1, info['KABKOT']
                      strtext = info['KABKOT']#.replace('JAKARTA PUSAT','JAKPUS').replace('JAKARTA SELATAN','JAKSEL').replace('JAKARTA UTARA','JAKUT').replace('JAKARTA TIMUR','JAKTIM').replace('JAKARTA BARAT','JAKBAR').replace('TANGERANG SELATAN','TANGSEL').replace('TANGERANG','TANG')
                      pl.text(x, y, strtext.replace(' ','\n'),fontsize=8)
                  # else:                      
                      # patches.append( Polygon(np.array(shape), True) )
        # ax = pl.gca()
        # ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2)) 
        m.readshapefile(shpprovfilename, 'PROV')
        patches = []
        for info, shape in zip(m.PROV_info, m.PROV):
            if info['Propinsi'] != 'SULAWESI TENGAH':
                patches.append( Polygon(np.array(shape), True) )
        ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2))
        
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Precipitation Anomaly mm')
    pl.title('Monthly Precipitation Anomaly - La Nina - ' + monthstr )
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show() 
    fig.savefig(dirout + domain + '-LaNina_monthly_anom_mm_' + str(mon+1) + '.png', format='png', dpi=300, bbox_inches='tight')

# plotting - monthly anomaly % - La Nina
for mon in range(12):
    monthstr = dt.date(1900, mon+1, 1).strftime('%B')

    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-100,101,10)
    month_anom_lanina_percent_tr = np.transpose(month_anom_lanina_percent[mon])
    if lsmask == True:
        month_anom_lanina_percent_tr = maskoceans(lon,lat,month_anom_lanina_percent_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, month_anom_lanina_percent_tr, clevs, extend = 'both', cmap=pl.cm.RdBu) 
    # Add Grid Lines
    m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    m.drawcountries()
    if kab == True: 
        m.readshapefile(shpkabkotfilename, 'KABKOT')
        patches = []
        for info, shape in zip(m.KABKOT_info, m.KABKOT):
              max = np.amax(shape, axis=0)
              min = np.amin(shape, axis=0)
              diff = max-min
              londistance = 2*diff[0]
              pol = shapely.geometry.Polygon(shape)
              center = pol.centroid.coords[0]
              centerdiff = center-min
              x1 = center[0]-0.28*centerdiff[0]
              y1 = center[1] 
              #x1 = min[0]+0.4*diff[0] #lon
              #y1 = min[1]+0.2*diff[0] #lat
              #mid = min+diff
              #print mid
              if x1>minlon+0.2 and x1<maxlon-0.2 and y1>minlat+0.2 and y1<maxlat-0.2 and londistance>0.5:
                  if info['PROVINSI'] == 'SULAWESI TENGAH' or info['PROVINSI'] == 'Sulawesi Tengah':
                      x, y = m(x1, y1)
                      #print y1, x1, info['KABKOT']
                      strtext = info['KABKOT']#.replace('JAKARTA PUSAT','JAKPUS').replace('JAKARTA SELATAN','JAKSEL').replace('JAKARTA UTARA','JAKUT').replace('JAKARTA TIMUR','JAKTIM').replace('JAKARTA BARAT','JAKBAR').replace('TANGERANG SELATAN','TANGSEL').replace('TANGERANG','TANG')
                      pl.text(x, y, strtext.replace(' ','\n'),fontsize=8)
                  # else:                      
                      # patches.append( Polygon(np.array(shape), True) )
        # ax = pl.gca()
        # ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2)) 
        m.readshapefile(shpprovfilename, 'PROV')
        patches = []
        for info, shape in zip(m.PROV_info, m.PROV):
            if info['Propinsi'] != 'SULAWESI TENGAH':
                patches.append( Polygon(np.array(shape), True) )
        ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2))
        
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Precipitation Anomaly %')
    pl.title('Monthly Precipitation Anomaly - La Nina - ' + monthstr )
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show() 
    fig.savefig(dirout + domain + '-LaNina_monthly_anom_percent_' + str(mon+1) + '.png', format='png', dpi=300, bbox_inches='tight')

# plotting - monthly anomaly mm - El Nino
for mon in range(12):
    monthstr = dt.date(1900, mon+1, 1).strftime('%B')

    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-100,101,10)
    month_anom_elnino_tr = np.transpose(month_anom_elnino[mon])
    if lsmask == True:
        month_anom_elnino_tr = maskoceans(lon,lat,month_anom_elnino_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, month_anom_elnino_tr, clevs, extend = 'both', cmap=pl.cm.RdBu) 
    # Add Grid Lines
    m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    m.drawcountries()
    if kab == True: 
        m.readshapefile(shpkabkotfilename, 'KABKOT')
        patches = []
        for info, shape in zip(m.KABKOT_info, m.KABKOT):
              max = np.amax(shape, axis=0)
              min = np.amin(shape, axis=0)
              diff = max-min
              londistance = 2*diff[0]
              pol = shapely.geometry.Polygon(shape)
              center = pol.centroid.coords[0]
              centerdiff = center-min
              x1 = center[0]-0.28*centerdiff[0]
              y1 = center[1] 
              #x1 = min[0]+0.4*diff[0] #lon
              #y1 = min[1]+0.2*diff[0] #lat
              #mid = min+diff
              #print mid
              if x1>minlon+0.2 and x1<maxlon-0.2 and y1>minlat+0.2 and y1<maxlat-0.2 and londistance>0.5:
                  if info['PROVINSI'] == 'SULAWESI TENGAH' or info['PROVINSI'] == 'Sulawesi Tengah':
                      x, y = m(x1, y1)
                      #print y1, x1, info['KABKOT']
                      strtext = info['KABKOT']#.replace('JAKARTA PUSAT','JAKPUS').replace('JAKARTA SELATAN','JAKSEL').replace('JAKARTA UTARA','JAKUT').replace('JAKARTA TIMUR','JAKTIM').replace('JAKARTA BARAT','JAKBAR').replace('TANGERANG SELATAN','TANGSEL').replace('TANGERANG','TANG')
                      pl.text(x, y, strtext.replace(' ','\n'),fontsize=8)
                  # else:                      
                      # patches.append( Polygon(np.array(shape), True) )
        # ax = pl.gca()
        # ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2)) 
        m.readshapefile(shpprovfilename, 'PROV')
        patches = []
        for info, shape in zip(m.PROV_info, m.PROV):
            if info['Propinsi'] != 'SULAWESI TENGAH':
                patches.append( Polygon(np.array(shape), True) )
        ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2))
        
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Precipitation Anomaly mm')
    pl.title('Monthly Precipitation Anomaly - El Nino - ' + monthstr )
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show() 
    fig.savefig(dirout + domain + '-ElNino_monthly_anom_mm_' + str(mon+1) + '.png', format='png', dpi=300, bbox_inches='tight')

# plotting - monthly anomaly % - El Nino
for mon in range(12):
    monthstr = dt.date(1900, mon+1, 1).strftime('%B')

    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-100,101,10)
    month_anom_elnino_percent_tr = np.transpose(month_anom_elnino_percent[mon])
    if lsmask == True:
        month_anom_elnino_percent_tr = maskoceans(lon,lat,month_anom_elnino_percent_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, month_anom_elnino_percent_tr, clevs, extend = 'both', cmap=pl.cm.RdBu) 
    # Add Grid Lines
    m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    m.drawcountries()
    if kab == True: 
        m.readshapefile(shpkabkotfilename, 'KABKOT')
        patches = []
        for info, shape in zip(m.KABKOT_info, m.KABKOT):
              max = np.amax(shape, axis=0)
              min = np.amin(shape, axis=0)
              diff = max-min
              londistance = 2*diff[0]
              pol = shapely.geometry.Polygon(shape)
              center = pol.centroid.coords[0]
              centerdiff = center-min
              x1 = center[0]-0.28*centerdiff[0]
              y1 = center[1] 
              #x1 = min[0]+0.4*diff[0] #lon
              #y1 = min[1]+0.2*diff[0] #lat
              #mid = min+diff
              #print mid
              if x1>minlon+0.2 and x1<maxlon-0.2 and y1>minlat+0.2 and y1<maxlat-0.2 and londistance>0.5:
                  if info['PROVINSI'] == 'SULAWESI TENGAH' or info['PROVINSI'] == 'Sulawesi Tengah':
                      x, y = m(x1, y1)
                      #print y1, x1, info['KABKOT']
                      strtext = info['KABKOT']#.replace('JAKARTA PUSAT','JAKPUS').replace('JAKARTA SELATAN','JAKSEL').replace('JAKARTA UTARA','JAKUT').replace('JAKARTA TIMUR','JAKTIM').replace('JAKARTA BARAT','JAKBAR').replace('TANGERANG SELATAN','TANGSEL').replace('TANGERANG','TANG')
                      pl.text(x, y, strtext.replace(' ','\n'),fontsize=8)
                  # else:                      
                      # patches.append( Polygon(np.array(shape), True) )
        # ax = pl.gca()
        # ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2)) 
        m.readshapefile(shpprovfilename, 'PROV')
        patches = []
        for info, shape in zip(m.PROV_info, m.PROV):
            if info['Propinsi'] != 'SULAWESI TENGAH':
                patches.append( Polygon(np.array(shape), True) )
        ax.add_collection(PatchCollection(patches, facecolor= 'white', edgecolor='k', linewidths=1., zorder=2))
        
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Precipitation Anomaly %')
    pl.title('Monthly Precipitation Anomaly - El Nino - ' + monthstr )
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show() 
    fig.savefig(dirout + domain + '-ElNino_monthly_anom_percent_' + str(mon+1) + '.png', format='png', dpi=300, bbox_inches='tight')

# ONI - plot correlation coeff. and p-values
fig = pl.figure(figsize=(10,8))
m = Basemap(projection="merc", resolution='h', \
              llcrnrlat=minlat,urcrnrlat=maxlat,\
                  llcrnrlon=minlon,urcrnrlon=maxlon,\
                      epsg=4238)
    
m.drawcoastlines(color="black") 
lon,lat = np.meshgrid(lons, lats) 
x,y = m(lon, lat)
clevs = np.arange(-1,1.1,0.1)
corr_oni_all_tr = np.transpose(corr_oni_all)
if lsmask == True:
    corr_oni_all_tr = maskoceans(lon,lat,corr_oni_all_tr,resolution='f',grid=1.25)
prec_plot = m.contourf(x, y, corr_oni_all_tr, clevs, extend = 'neither', cmap=pl.cm.RdBu.reversed())
# m.contour(x, y, np.transpose(pval_oni_all))
# Add Grid Lines
# m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
# m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
cb.set_label('Correlation Coefficient')
pl.title('Oceanic Nino Index vs. Precipitation - GPM_IMERGV06 (2001-2019)')
# plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
pl.show()
fig.savefig(dirout + domain +'-corr_coef_oni-all.png', format='png', dpi=300, bbox_inches='tight')

# ONI - plot seasonal
seasonstr = ('DJF', 'MAM', 'JJA', 'SON')
for season in range(4):
    fig = pl.figure(figsize=(10,8))
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-1,1.1,0.1)
    corr_oni_season_tr = np.transpose(corr_oni_season[season,:,:])
    if lsmask == True:
        corr_oni_season_tr = maskoceans(lon,lat,corr_oni_season_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, corr_oni_season_tr, clevs, extend = 'neither', cmap=pl.cm.RdBu.reversed())
    # m.contour(x, y, np.transpose(pval_oni_season[season,:,:])
    # Add Grid Lines
    # m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    # m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Correlation Coefficient')
    pl.title('Oceanic Nino Index vs. Precipitation - GPM_IMERGV06 (2001-2019) - ' + seasonstr[season])
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show()
    fig.savefig(dirout + domain +'-corr_coef_oni_' + seasonstr[season] + '.png', format='png', dpi=300, bbox_inches='tight')

# ONI - plot monthly
for mon in range(12):
    monthstr = dt.date(1900, mon+1, 1).strftime('%B')
    fig = pl.figure(figsize=(10,8))
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-1,1.1,0.1)
    corr_oni_month_tr = np.transpose(corr_oni_month[mon,:,:])
    if lsmask == True:
        corr_oni_month_tr = maskoceans(lon,lat,corr_oni_month_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, corr_oni_month_tr, clevs, extend = 'neither', cmap=pl.cm.RdBu.reversed())
    # m.contour(x, y, np.transpose(pval_oni_season[season,:,:])
    # Add Grid Lines
    # m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    # m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Correlation Coefficient')
    pl.title('Oceanic Nino Index vs. Precipitation - GPM_IMERGV06 (2001-2019) - ' + monthstr)
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show()
    fig.savefig(dirout + domain +'-corr_coef_oni_' + str(mon+1) + '.png', format='png', dpi=300, bbox_inches='tight')

# NINO34 - plot correlation coeff. and p-values
fig = pl.figure(figsize=(10,8))
m = Basemap(projection="merc", resolution='h', \
              llcrnrlat=minlat,urcrnrlat=maxlat,\
                  llcrnrlon=minlon,urcrnrlon=maxlon,\
                      epsg=4238)
    
m.drawcoastlines(color="black") 
lon,lat = np.meshgrid(lons, lats) 
x,y = m(lon, lat)
clevs = np.arange(-1,1.1,0.1)
corr_nino34_all_tr = np.transpose(corr_nino34_all)
if lsmask == True:
    corr_nino34_all_tr = maskoceans(lon,lat,corr_nino34_all_tr,resolution='f',grid=1.25)
prec_plot = m.contourf(x, y, corr_nino34_all_tr, clevs, extend = 'neither', cmap=pl.cm.RdBu.reversed())
# m.contour(x, y, np.transpose(pval_nino34_all))
# Add Grid Lines
# m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
# m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
cb.set_label('Correlation Coefficient')
pl.title('NINO3.4 vs. Precipitation - GPM_IMERGV06 (2001-2019)')
# plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
pl.show()
fig.savefig(dirout + domain +'-corr_coef_nino34-all.png', format='png', dpi=300, bbox_inches='tight')

# NINO34 - plot seasonal
seasonstr = ('DJF', 'MAM', 'JJA', 'SON')
for season in range(4):
    fig = pl.figure(figsize=(10,8))
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-1,1.1,0.1)
    corr_nino34_season_tr = np.transpose(corr_nino34_season[season,:,:])
    if lsmask == True:
        corr_nino34_season_tr = maskoceans(lon,lat,corr_nino34_season_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, corr_nino34_season_tr, clevs, extend = 'neither', cmap=pl.cm.RdBu.reversed())
    # m.contour(x, y, np.transpose(pval_nino34_season[season,:,:])
    # Add Grid Lines
    # m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    # m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Correlation Coefficient')
    pl.title('NINO3.4 vs. Precipitation - GPM_IMERGV06 (2001-2019) - ' + seasonstr[season])
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show()
    fig.savefig(dirout + domain +'-corr_coef_nino34_' + seasonstr[season] + '.png', format='png', dpi=300, bbox_inches='tight')

# NINO34 - plot monthly
for mon in range(12):
    monthstr = dt.date(1900, mon+1, 1).strftime('%B')
    fig = pl.figure(figsize=(10,8))
    m = Basemap(projection="merc", resolution='h', \
                  llcrnrlat=minlat,urcrnrlat=maxlat,\
                      llcrnrlon=minlon,urcrnrlon=maxlon,\
                          epsg=4238)
        
    m.drawcoastlines(color="black") 
    lon,lat = np.meshgrid(lons, lats) 
    x,y = m(lon, lat)
    clevs = np.arange(-1,1.1,0.1)
    corr_nino34_month_tr = np.transpose(corr_nino34_month[mon,:,:])
    if lsmask == True:
        corr_nino34_month_tr = maskoceans(lon,lat,corr_nino34_month_tr,resolution='f',grid=1.25)
    prec_plot = m.contourf(x, y, corr_nino34_month_tr, clevs, extend = 'neither', cmap=pl.cm.RdBu.reversed())
    # m.contour(x, y, np.transpose(pval_nino34_season[season,:,:])
    # Add Grid Lines
    # m.drawparallels(np.arange(math.floor(minlat),math.ceil(maxlat),5.0),labels=[1,0,0,0],linewidth=1, fontsize=10)
    # m.drawmeridians(np.arange(math.floor(minlon),math.ceil(maxlon),5.0),labels=[0,0,0,1],linewidth=1, fontsize=10)
    cb = m.colorbar(prec_plot, "right", ticks=clevs, size="5%", pad="2%")
    cb.set_label('Correlation Coefficient')
    pl.title('NINO3.4 vs. Precipitation - GPM_IMERGV06 (2001-2019) - ' + monthstr)
    # plt.annotate('Data - CRU TS v4.02',(-178,-88), fontsize=6)
    pl.show()
    fig.savefig(dirout + domain +'-corr_coef_nino34_' + str(mon+1) + '.png', format='png', dpi=300, bbox_inches='tight')
