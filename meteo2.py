#!/usr/bin/env python
# coding: utf-8
# encoding=utf-8

from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
import os, shutil                         # For issuing commands to the OS.
from os import listdir
from os.path import isfile, join
import time
import random
from matplotlib.font_manager import FontProperties
import pandas as pd
import csv
from scipy import stats
import matplotlib.pyplot as plt
import datetime as dt
import datetime
from scipy.optimize import curve_fit
from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#from sklearn.metrics import confusion_matrixfrom sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from io import BytesIO

import boto3
import urllib
import netCDF4 as nc
from netCDF4 import MFDataset
import urllib.request
from time import sleep
import requests
from netCDF4 import Dataset
import timeit

# ggap:
#   t
#	p = 37
#	latitude = 256
#	longitude = 512
# p = 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600,
#    550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50,
#    30, 20, 10, 7, 5, 3, 2, 1 ;

# I calculated that 4-8 miles atm pressure height corresponds to 440 - 160 mb, so that we need to take the pressure levels
# p = 400, 350, 300, 250, 200, 175, 150 (roughly), which means p[17:24]. (24 not included.. how is it in Python again?)


#lat [ 89.462944    88.766945    88.06699     87.366035    86.6648
#  85.96337     85.26186     84.56026     83.858635    83.15699
#  82.455315    81.75364     81.05194     80.350235    79.64852 ....
#  ....78.9468      78.245094    77.543365    76.84162     76.139915
#   8.771913     8.070159     7.368407     6.6666536    5.964901
#   5.2631474    4.5613947    3.859641     3.1578884    2.4561355
#   1.7543825    1.0526295    0.35087648  -0.35087648

# http://www.met.reading.ac.uk/~marc/it/snap/varList/eraVars/  --- list of variables

gafs = 'gafs01051979/gafs197901050003.nc' # 00/12 + 3 6 9 12
ggap = 'ggap01051979/ggap197901050000-c.nc' # 00, 06, 12, 18

dir = '/Volumes/Seagate Backup Plus Drive/meteo-badc/ggap/'
# 1979-1993 (1995-2003)
year = '2003/' # how to list content of directory?

dir1 = '/Volumes/Seagate Backup Plus Drive/meteo-2/ggap/'
# 1994-2018
# disk1

year_n = 2003 #1979

def name_is_good(f):
    if f[-3:] == '.nc':
        return True
    else:
        False

# constants needed for arc calculation:
R = 6371 * 1e3  # Earth radius, in m
C = 2 * np.pi * R / 512.  # dr = 2*pi*R*cos(pi*phi/180) would be the whole arc

#-----------------------------------------------------------------------------------------------------------------------


def circulation(p0,p1,l0,l1,v,lat):

    # calculate the length of the arc between two grid points. This quantity depends on the latitude

    #print('v.shape',np.shape(v))
    #print('pressure height of interest (in mb)',pa,len(pa))

    lat_selected = np.array([lat[i] for i in range(l0, l1)])

    arc = np.zeros((p1 - p0, len(lat_selected)))  # the radius for the calculation depends on the latitude
    cir = np.zeros((p1 - p0, len(lat_selected)))  # !!! an error ocurred bc of non restriction of latitudes in the initialization

    #print('shape of circulation',np.shape(cir))
    #print('nb of pressure height elements in the interval',p0,p1,':',p1-p0,'=?')

    for la in range(len(lat_selected)):
        r = np.cos(np.pi / 180 * lat_selected[la])  # latitude array in degrees. 'flat radius' (radius of the cut circle at given lat) corresponding to the given latitude
        arc[:,la] = C * r  # calculation of a little piece of the Umfang (line length) of the latitude arc that goes around the globe.

    cir[:,:] = np.sum(v[0, p0:p1, l0:l1, :],axis=2) * arc[:,:] * 1.0e-6  # asum over longitudinal points,

        #print('min, mean, max daily mean circulation per day [km2/s]', cir.min(), cir.mean(), cir.max())

    return cir,lat_selected

def max_circ(cir,p0,l0):

    #print(np.shape(cir))

    maxim = np.unravel_index(cir[:,:].argmax(), cir[:,:].shape)

    (pre, latit) = maxim

    pre_init = pre + p0
    lat_init = latit + l0

    return maxim,pre_init,lat_init

def adjustFigAspect(fig, aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
               aspect ratio.
    '''
    xsize, ysize = fig.get_size_inches()
    minsize = min(xsize, ysize)
    xlim = .4 * minsize / xsize
    ylim = .4 * minsize / ysize
    if aspect < 1:
        ylim *= aspect
    else:
        xlim /= aspect
    fig.subplots_adjust(left=.5 - xlim,
                        right=.5 + xlim,
                        bottom=.5 - ylim,
                        top=.5 + ylim)





def plot(timestep,p0,p1,l0, l1,cir,year_n):
    plt.switch_backend('agg')
    cir_min = -1500
    cir_max = 1500
    levels = MaxNLocator(nbins=15).tick_values(cir[:, :].min(), cir[:, :].max())
    # fig, (ax0, ax1) = plt.subplots(nrows=2)
    fig, ax1 = plt.subplots(nrows=1)
    im = plt.imshow(np.transpose(cir[:, :]), extent=(p0 * 10, (p1 - 1) * 10, l1 - 1, l0),
                    interpolation='nearest', cmap=cm.gist_rainbow)
    # ax1.set_xticks(range(len(labels1)), [i for i in labels1])
    # ax = fig.add_subplot(131)
    # mesh = ax1.pcolormesh(data, cmap=cm)
    im.set_clim(cir_min, cir_max)
    fig.colorbar(im, shrink=0.6)
    ax1.set_title('daily wind circulation (' + str(year_n) + '), day %03d ' % (timestep))
    ax1.set_xlabel('pressure height')
    ax1.set_ylabel('latitudes')
    fig.tight_layout()
    print('time step', timestep)
    plt.savefig("_tmp%05d.png" % timestep)




def plot_final(p0,p1,l0,l1,cir,var,color):
    fh = 5  # figheight
    fw = 12.5  # figwidth

    m = 6
    plt.switch_backend('agg')

    #fig = plt.figure()
    #levels = MaxNLocator(nbins=15).tick_values(cir[:, :].min(), cir[:, :].max())
    #fig, (ax0, ax1) = plt.subplots(nrows=2)

    #ax1 = fig.add_subplot(311)
    fig, ax1 = plt.subplots(nrows=1)
    #adjustFigAspect(fig, aspect=.4)

    fig.set_figheight(fh)
    fig.set_figwidth(fw)

    # y0 = 1979
    #xtim = np.array([y0 + i for i in range(40)])


    plt.plot((cir),'o',color,linewidth=1.1)
    ax1.set_title('daily '+str(var)+' for region (p,lat) = (['+str(p0)+','+str(p1)+'],['+str(l0)+','+str(l1)+'])')
    ax1.set_xlabel('year')
    ax1.set_ylabel(''+str(var)+'')

    fig.tight_layout()
    plt.xticks(fontsize=m)
    plt.yticks(fontsize=m)
    ax1.legend(fontsize=m)

    plt.savefig(''+str(var)+'1979-2018.png')



def Read_files(mypath,files_for_day):

    v_total = []
    for file in files_for_day:  # daily values

        # ---------------------------------------------------------------------------------------------------
        # 1. Read in the (wind) data
        # ----------------------------------------------------------------------------------------------------

        f = Dataset(mypath + file, 'r')

        # print('read in file ', file)
        p = f.variables['p'][:]
        lat = f.variables['latitude'][:]
        #lon = f.variables['longitude'][:]
        # vort = f.variables['VO'] # vorticity [1/s]
        v = f.variables['U']  # zonal wind (t,p,lat,lon)
        # time = f.variables['t'][:] # in these files this is only 1 value! (daily)
        # print('finished reading in file ', file)

        # calculate daily mean of v----:
        v_total.append(v)  # (t,p,lat,lon),(t,p,lat,lon),(t,p,lat,lon),...

    v_total = np.asarray(v_total)
    v_total = np.sum(v_total, axis=0) / len(files_for_day)

    return v_total,lat,p

def Main(files_for_day,mypath,p0, p1, l0, l1,timestep,plo):


        v_total,lat,p = Read_files(mypath,files_for_day)

        # -----------------------------------------------------------------------------------------------------------------------
        # 2. Calculate the circulation along a latitude circle around the globe: Sum_over_longitudes(v)*dl
        # ---------------------------------------------------------------------------------------------------

        cir,lat_selected = circulation(p0, p1, l0, l1, v_total, lat)

        if plo == True:
           plot(timestep, p0, p1, l0, l1,cir,year_n)


        # -----------------------------------------------------------------------------------------------------------------------        #
        # 3. Calculate maximum circulation in given region for given day
        # -----------------------------------------------------------------------------------------------------------------------

        maxi,pre_init,lat_init = max_circ(cir,p0,l0)
        (pre, latit) = maxi

        #print('shape of circ',np.shape(cir),'cir[',pre_init, lat_init,']', cir[pre, latit], 'p', p[pre_init], ' (mb), lat', lat[lat_init],' (degrees) for file',file,'day ',timestep)

        #np.where(a == a.max())
        # (array([0]), array([2]))
        return cir, maxi, pre_init,lat_init

#-----------------------------------------------------------------------------------------------------------------------
#
#
# Main code
#
#
#-----------------------------------------------------------------------------------------------------------------------
# files in year .. files from path
# dates...         strings corresponding to indices 4:12 (YYYYMMDD) in the filenames
# mypath ... mypath = dir + year

# prepare the files that we want to read in
# the file names are of the kind ggapYYYYMMDD-c.nc. Determine a list of files corresponding to a same day
# then for each day, we can read in the files and calculate the daily mean


# there are 37 pressure height levels
pr0 =  15 #7 # 400mb
pr1 =  25 # 15
# 100mb
# define a geographic region ('rectangular')-------------------------------------------------------------------------
lat0 = 150  # 5 # input in degrees N, must be more southward than lat1
lat1 = 200# 60  # 90


start = timeit.default_timer()
fps = 3
#day = 110
#d = dates[day] # has 365 inputs because of np.unique only takes sorted unique value...same day :-)
# list of files for this given day:
#files_for_day = [f for f in files_in_year if f[4:12] == d]

# 1979-1993
year0 = 1979
yrs = [(str(year0+i)+'/') for i in range(15)]
yrs_ = [(year0+i) for i in range(15)]
# 1994-2018, in combination with a different hard disk and path
yrs1 = [(str(year0+i)+'/') for i in range(15,40)]
yrs1_ = [(year0+i) for i in range(15,40)]

print(yrs1)
print(yrs1_)

#mypath = dir + year

# dir1 if we use the other hard disk ()
mypath = [dir1 + ye for ye in yrs1]
calculate_circulation = False
if calculate_circulation == True:

 for k in range(len(yrs1_)):

    mypaths = dir1 + yrs1[k]

    files_in_year = [f for f in listdir(mypaths) if isfile(join(mypaths, f)) and name_is_good(f)]  # lists files from path
    dates = np.unique(np.array([f[4:12] for f in files_in_year]))  # dates: strings corresponding to indices 4:12 in the filenames
    dates2 = [dates[d] for d in range(len(dates)) if d % 5 == 0]

    print('year ', yrs1_[k])
    timestep = 0

    file_name = 'Circ_'+str(yrs1_[k])+'.txt'
    #with open('Circ_'+str(yrs1_[k])+'.txt',w+) as outfile:
    f = open(file_name,'w+')
    #with open(newfilename1, 'w+') as outfile:
    #        np.savetxt(outfile, ar, fmt=['%10.6f', '%10.8f'])
    #    dec_data = np.genfromtxt(newfilename, usecols=(0, 1))

    #print(len(dates))
    for d in dates:  # all files, sorted according to the date, for a given year. Each file corresponds to a year
#
        files_for_day = [f for f in files_in_year if f[4:12] == d]  # list of files corresponding to the given day d in the loop

        cir, max_cir, pre, latit = Main(files_for_day,mypaths,pr0, pr1, lat0, lat1,timestep,False)

        #cir.append(cir) # ,pre.append(), latit
        print('write to file for d',d)
        f.write('%6.3f\t %d\t %d\t \n' % (cir[max_cir],pre,latit))  # \n.. next line


        # L =\t%f \n
        #np.savetxt(outfile, ar, fmt=['%10.6f', '%10.8f'])
        #np.savetxt('circulation_year'+str(yrs1_[k])+'.txt',cir) # save circulation (pr,lat) in a file for each ...day and year

        timestep = timestep + 1

    f.close()

#------- final analysis of the time series---------------------

year_list = [(year0+i) for i in range(40)]
print('analysis for years', year_list)





total_cir = [np.loadtxt('Circ_'+str(j)+'.txt') for j in year_list] #- result is [nparray(365,3) , nparray(366,3) , ...]

print('shape',[np.shape(x) for x in total_cir])

total_cir = np.concatenate(total_cir,axis=0)

print(np.shape(total_cir))

# plot
plot_final(pr0,pr1,lat0,lat1,total_cir[:, 0],'circulation','b')
plot_final(pr0,pr1,lat0,lat1,total_cir[:, 1],'pressure height','g')
plot_final(pr0,pr1,lat0,lat1,total_cir[:, 2],'latitude','r')
exit()

#---------------------------------------------------------



    #print(np.shape(cir)) # cir has shape (pr,lat)
        #exit()
        #with file('Yearlydeviation'+str(region)+'.txt','w') as outfile:
        #    outfile.write('# Array shape of yearly T mean:{0}\n'.format(summer_dev.shape))
        #    for slice_tot in summer_dev:
        #        np.savetxt(outfile,slice_tot)
        #        outfile.write('# Newline\n')
        # summer_mean = np.loadtxt('circulation_year'+str(yrs_[k])+'.txt')
        # summer_mean = summer_mean.reshape((pr1-pr0,lat1-lat0))
        #np.zeros((p1 - p0, len(lat_selected)))


#np.savetxt('Models'+str(region)+'.txt',models)
# print '--- CREATE TEXTFILE FOR YEARLY DEVIATION'
#with file('Yearlydeviation'+str(region)+'.txt','w') as outfile:
#    outfile.write('# Array shape of yearly T mean:{0}\n'.format(summer_dev.shape))
#    for slice_tot in summer_dev:
#        np.savetxt(outfile,slice_tot)
#        outfile.write('# Newline\n')

#summer_mean = np.loadtxt('Yearlymean'+str(region)+'.txt')
#summer_mean = summer_mean.reshape((len(models),16,nb_of_years))
#summer_dev = np.loadtxt('Yearlydeviation'+str(region)+'.txt')
#summer_dev = summer_dev.reshape((len(models),16,nb_of_years))






    #savefig = plot(timestep,pr0,pr1,lat0,lat1,cir,year_n) # timestep,p0,p1,l0, l1,cir,year_n)

    #if timestep == int(365/5.):
    #    os.system("rm -f movie.mp4")
    #    os.system("./ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie.mp4")
    #    os.system("rm _tmp*.png")
    #    exit()
#    timestep = timestep + 1

#exit()
#---------------------------test program
#print(files_for_day)

# v,lat,p = Read_files(mypath,['ggap197904210600-c.nc'])
#
# ci,lat_selected = circulation(0,37,0,25,v,lat)
#
# print('cir',ci[0,0])

#cir, pre, latit = Main(files_for_day,mypath,pr0, pr1, lat0, lat1,day)

#pr0 = 20
##pr1=  25
#lat0= 50
#lat1= 100

#cir, max_cir, pre, latit = Main(['ggap197904210600-c.nc'],mypath,pr0, pr1, lat0, lat1, day)

#print('max circ',cir[max_cir],"max_cir",max_cir, 'at indices (p,l)=',pre, latit, 'for (pr0,pr1)=',pr0,pr1,' (l0,l1)=',lat0,lat1)

#print(d)

#exit()








        #-----------------------------------------------------------------------------------------------------------------------
        # 4. Calculate the Laplacian and the first derivative of the circulation function as a function
        #    of p and latitude (at each grid point except at the boundaries)
        #
        #   Delta d2f = d2f/dlat2 + d2f/dp2, df = df_p + df_l
        #-----------------------------------------------------------------------------------------------------------------------
#         df2 = np.zeros((p1 - p0, len(lat)))
#         df = np.zeros((p1 - p0, len(lat)))
#         k = 1
#         for pr in range(k,len(pa)-k): # indices of pressure. stops at 23, 24 should not be in the list
#             for l in range(k,len(lat)-k):
#                 #       if pr-1 >= 0 and pr+1 <= p1-p0-2:
#                 d2f_p = (cir[pr+k,l]+cir[pr-k,l]-2*cir[pr,l])/float(pa[pr+k]-pa[pr])
#                 df_p =  (cir[pr+k,l]-cir[pr-k,l])/(2.*(pa[pr+k]-pa[pr]))
#                 d2f_l = (cir[pr, l+k] + cir[pr, l-k]-2*cir[pr,l])/float(lat[l+k]-lat[l])
#                 df_l = (cir[pr, l+k] - cir[pr, l-k]) /(2. * (lat[l+k]-lat[l]))
#                 # scale the Laplacian by the norm of the first derivative
#                 df2[pr,l] = (d2f_p + d2f_l)/np.sqrt(abs(df_p)**2 +abs(df_l)**2)
#                 df[pr, l] = np.sqrt(abs(df_p)**2 +abs(df_l)**2)
#
#
#         # -----------------------------------------------------------------------------------------------------------------------
#         # 2D plot
#         # -----------------------------------------------------------------------------------------------------------------------
#
#         timestep = timestep + 1
#
#         if timestep == 5:
#             plt.switch_backend('agg')
#             fig = plt.figure()
#             levels = MaxNLocator(nbins=15).tick_values(cir[:, :].min(), cir[:, :].max())
#             # pick the desired colormap, sensible levels, and define a normalization
#             # instance which takes data values and translates those into levels.
#             cmap = plt.get_cmap('PiYG')
#             norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#             # fig, (ax0, ax1) = plt.subplots(nrows=2)
#             fig, ax1 = plt.subplots(nrows=1)
#
#             pa = np.array([i for i in range(p0, p1)])  # array indices of pressure height
#
#             x = np.array([p[prs] for prs in pa])  # pressure at index, in mbar
#
#             y = lat  # latitude in degrees
#
#             im = plt.imshow(cir[:, :], extent=(x.min(), x.max(), y.max(), y.min()),
#                             interpolation='nearest', cmap=cm.gist_rainbow)
#
#
#             fig.colorbar(im, shrink=0.4)
#             ax1.set_title('circulation')
#             fig.tight_layout()
#             plt.savefig('test_2D_circul'+str(timestep)+'.pdf')
#
#
#             # 2D plot
#
#             plt.switch_backend('agg')
#             fig = plt.figure()
#             levels = MaxNLocator(nbins=15).tick_values(df2[:, :].min(), df2[:, :].max())
#
#
#             # pick the desired colormap, sensible levels, and define a normalization
#             # instance which takes data values and translates those into levels.
#             cmap = plt.get_cmap('PiYG')
#             norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#             # fig, (ax0, ax1) = plt.subplots(nrows=2)
#             fig, ax1 = plt.subplots(nrows=1)
#             pa = np.array([i for i in range(p0, p1)])  # array indices of pressure height
#             x = np.array([p[prs] for prs in pa])  # pressure at index, in mbar
#             y = lat  # latitude in degrees
#             im = plt.imshow(df2[:, :], extent=(x.min(), x.max(), y.max(), y.min()),
#                             interpolation='nearest', cmap=cm.gist_rainbow)
#             fig.colorbar(im, shrink=0.4)
#             ax1.set_title('Laplacian of circulation')
#             fig.tight_layout()
#             plt.savefig('test_2D_Laplace'+str(timestep)+'.pdf')
#
#
#             plt.switch_backend('agg')
#             fig = plt.figure()
#             levels = MaxNLocator(nbins=15).tick_values(df[:, :].min(), df[:, :].max())
#             # pick the desired colormap, sensible levels, and define a normalization
#             # instance which takes data values and translates those into levels.
#             cmap = plt.get_cmap('PiYG')
#             norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#             # fig, (ax0, ax1) = plt.subplots(nrows=2)
#             fig, ax1 = plt.subplots(nrows=1)
#             pa = np.array([i for i in range(p0, p1)])  # array indices of pressure height
#             x = np.array([p[prs] for prs in pa])  # pressure at index, in mbar
#             y = lat  # latitude in degrees
#
#             #im = plt.imshow(df[:, :], extent=(x.min(), x.max(), y.max(), y.min()),
#             #                interpolation='nearest', cmap=cm.gist_rainbow)
#
#             im = plt.imshow(df[:, :], extent=(x.min(), x.max(), y.max(), y.min()),
#                             interpolation='nearest', cmap=cm.gist_rainbow)
#             fig.colorbar(im, shrink=0.4)
#             ax1.set_title('Gradient of circulation')
#             fig.tight_layout()
#             plt.savefig('test_2D_gradient'+str(timestep)+'.pdf')
#
#
# exit()
#     # these values in 3 and 2 are daily means
#
#
#
#     #


movie = False
if movie == True:
 timestep = 0
 for d in dates:  # for those files corresponding to a same day, within each year, list all of them

    print('plot figure for time step',timestep)
    fps = 3                                         # nb of frames per second for the movie

    cir_min = -1500
    cir_max = 1500
    movie_figures = True
    if movie_figures == True:

     plt.switch_backend('agg')
     fig = plt.figure()
     levels = MaxNLocator(nbins=15).tick_values(cir[:, :].min(), cir[:, :].max())
     # pick the desired colormap, sensible levels, and define a normalization
     # instance which takes data values and translates those into levels.
     cmap = plt.get_cmap('PiYG')
     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
     # fig, (ax0, ax1) = plt.subplots(nrows=2)
     fig, ax1 = plt.subplots(nrows=1)
     pa = np.array([i for i in range(p0, p1)])  # array indices of pressure height
     x = np.array([p[prs] for prs in pa])  # pressure at index, in mbar
     y = lat  # latitude in degrees
     im = plt.imshow(cir[:, :], extent=(x.min(), x.max(), y.max(), y.min()),
                    interpolation='nearest', cmap=cm.gist_rainbow)
     fig.colorbar(im, shrink=0.4)



     #fig = plt.figure()
     #ax = fig.add_subplot(131)
     #mesh = ax1.pcolormesh(data, cmap=cm)
     im.set_clim(cir_min, cir_max)

     ax1.set_title('daily wind circulation (1979), day %03d ' %timestep)
     fig.tight_layout()
     print('time step',timestep)

     plt.savefig("_tmp%05d.png" % timestep)

     timestep = timestep +1
     #plt.clf()  # Clear the figure to make way for the next image.

     if timestep == 364:
        os.system("rm -f movie.mp4")
        os.system("./ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie.mp4")
        os.system("rm _tmp*.png")

        exit()




exit()






print_values = False
if print_values == True:
   for pr in range(len(pa)):  # indices of pressure. 24 should not be in the list
        pp = pa[pr]
        print('PRESSURE grid',pa[pr],'corresponding to (mb)',p[pp])
        print('-------------------')
        for l in range(len(lat)):  # l should be replaced, in v, by lat_grids[l]
           print('latitude',lat[l],'circulation = ',cir[0,pr,l])
        print(print('min,mean,max circ [km2/s]', np.min(cir[pr,:]), np.mean(cir[pr,:]), np.max(cir[pr,:]))) # for each pressure level calculate mean, min, max circulation



stop = timeit.default_timer()
print('Time (in min): ', (stop - start)/60.)



# this list enables us to calculate the daily mean of circulation




exit()


movie_figures = True


print(files_in_year)
#exit()



# list of all days:



timestep = 0

for file_ggap in files_in_year:

    #---------------------------------------------------------------------------------------------------
    # 1. read in the (wind) data
    #----------------------------------------------------------------------------------------------------
    f = Dataset(mypath + file_ggap,'r')

    print('read in file ',file_ggap,' and time step',timestep)
    p = f.variables['p'][:]
    lat = f.variables['latitude'][:]
    lon = f.variables['longitude'][:]
    #vort = f.variables['VO'] # vorticity [1/s]
    v = f.variables['U']     # zonal wind (t,p,lat,lon)
    #time = f.variables['t'][:] # in these files this is only 1 value! (daily)

    print('finished reading in file ',file_ggap)
    # 2. define a geographic region ('rectangular')-------------------------------------------------------------------------
    lat0 = -90 # 5 # input in degrees N, must be more southward than lat1
    lat1 = 90 # 90

    #lon0 = 5# if West
    #lon1 = 0 # if only west
    #print('lat 0, 1 (in degrees N)',lat0, lat1)


    #-----------------------------------------------------------------------------------------------------------------------
    # 3. Calculate the circulation along a latitude circle around the globe
    #
    #    I calculate, for each pressure height level, the sum, over all longitudinal grid points (lon = 1,512
    #    (whole longitude array!)), of the product taken at the given latitude: v(lon) * (lon_grid_length)|
    #    converted in (approx) m
    #---------------------------------------------------------------------------------------------------

    # calculate the length of the arc between two grid points. This quantity depends on the latitude.
    R = 6371*1e3 # Earth radius, in m
    # dr = 2*pi*R*cos(pi*phi/180) would be the whole arc
    arc = np.zeros(len(lat)) # the radius for the calculation depends on the latitude
    C = 2*np.pi*R/512.
    for la in range(len(lat)):
        # only for one latitude for now (in degrees N)
        r = np.cos(np.pi/180*lat[la]) # latitude array in degrees. 'flat radius' (radius of the cut circle at given lat) corresponding to the given latitude
        arc[la] = C*r #calculation of a little piece of the Umfang (line length) of the latitude arc that goes around the globe.
    # define the pressure height values we are interested in

    p0 = 5
    p1 = len(p)-3 # there are 37 pressure height levels

    #print('len p',len(p))
    start = timeit.default_timer()
    # calculate circulation
    cir = np.zeros((len(time),p1-p0,len(lat))) # !!! an error ocurred bc of non restriction of latitudes in the initialization
    pa = np.array([i for i in range(p0,p1)]) # pressure coordinate indices (grid pts of array)
    #print('pressure height of interest (in mb)',pa,len(pa))
    #for ti in range(len(time)):
    for pr in range(len(pa)): # indices of pressure. 24 should not be in the list
         for l in range(len(lat)): # l should be replaced, in v, by lat_grids[l]
             cir[ti,pr,l] = np.sum(v[ti,pa[pr],l,:])*arc[l]*1.0e-6 # asum over longitudinal points,

    stop = timeit.default_timer()
    print('Time (in min): ', (stop - start)/60.)

    #print('circulation',cir[0,0,:])
    print('min, mean, max circulation [in km2/s]', cir.min(), cir.mean(), cir.max())

    print_values = False
    if print_values == True:
     for pr in range(len(pa)):  # indices of pressure. 24 should not be in the list
        pp = pa[pr]
        print('PRESSURE grid',pa[pr],'corresponding to (mb)',p[pp])
        print('-------------------')
        for l in range(len(lat)):  # l should be replaced, in v, by lat_grids[l]
           print('latitude',lat[l],'circulation = ',cir[0,pr,l])
        print(print('min,mean,max circ [km2/s]', np.min(cir[0,pr,:]), np.mean(cir[0,pr,:]), np.max(cir[0,pr,:]))) # for each pressure level calculate mean, min, max circulation

    #-----------------------------------------------------------------------------------------------------------------------
    # 4. Calculate the Laplacian at each grid point except at the boundaries
    #
    #   Delta f = d2f/dx2 + d2f/
    #-----------------------------------------------------------------------------------------------------------------------
    # movie of 2D:


    print('plot figure for time step',timestep)
    fps = 2                                         # nb of frames per second for the movie

    if movie_figures == True:

     plt.switch_backend('agg')
     fig = plt.figure()
     levels = MaxNLocator(nbins=15).tick_values(cir[:, :].min(), cir[:, :].max())
     # pick the desired colormap, sensible levels, and define a normalization
     # instance which takes data values and translates those into levels.
     cmap = plt.get_cmap('PiYG')
     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
     # fig, (ax0, ax1) = plt.subplots(nrows=2)
     fig, ax1 = plt.subplots(nrows=1)
     pa = np.array([i for i in range(p0, p1)])  # array indices of pressure height
     x = np.array([p[prs] for prs in pa])  # pressure at index, in mbar
     y = lat  # latitude in degrees
     im = plt.imshow(cir[0, :, :], extent=(x.min(), x.max(), y.max(), y.min()),
                    interpolation='nearest', cmap=cm.gist_rainbow)
     fig.colorbar(im, shrink=0.4)
     ax1.set_title('wind circulationon (km2/s)')
     fig.tight_layout()
     print('time step',timestep)
     plt.savefig("_tmp%05d.png" % timestep)

     timestep = timestep +1
     #plt.clf()  # Clear the figure to make way for the next image.

     if timestep == 40:
        os.system("rm -f movie.mp4")
        os.system("./ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie.mp4")
        os.system("rm _tmp*.png")

        exit()


#-----------------------------------------------------------------------------------------------------------------------
exit()
# 2D plot

plt.switch_backend('agg')
fig = plt.figure()
levels = MaxNLocator(nbins=15).tick_values(cir[0,:,:].min(), cir[0,:,:].max())
# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#fig, (ax0, ax1) = plt.subplots(nrows=2)
fig, ax1 = plt.subplots(nrows=1)
pa = np.array([i for i in range(p0,p1)]) # array indices of pressure height
x = np.array([p[prs] for prs in pa])     # pressure at index, in mbar
y = lat                                  # latitude in degrees
im = plt.imshow(cir[0,:,:], extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gist_rainbow)
fig.colorbar(im,shrink = 0.4)
ax1.set_title('contourf with levels')
fig.tight_layout()
plt.savefig('test_2D_.pdf')



# 2D plot
#plt.switch_backend('agg')
#fig = plt.figure()
#levels = MaxNLocator(nbins=15).tick_values(cir[0,:,:].min(), cir[0,:,:].max())
## pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
#cmap = plt.get_cmap('PiYG')
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#fig, (ax0, ax1) = plt.subplots(nrows=2)
#pa = np.array([i for i in range(p0,p1)]) # array indices of pressure height
#x = np.array([p[prs] for prs in pa])     # pressure at index, in mbar
#y = lat                                  # latitude in degrees
#im = ax0.pcolormesh(x, y, np.transpose(cir[0,:,:]), cmap=cmap, norm=norm)
#fig.colorbar(im, ax=ax0)
#ax0.set_title('pcolormesh with levels')
## contours are *point* based plots, so convert our bound into point
## centers
#cf = ax1.contourf(x,y, np.transpose(cir[0,:,:]), levels=levels,cmap=cmap)
#fig.colorbar(cf, ax=ax1)
#ax1.set_title('contourf with levels')
# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
#fig.tight_layout()
#plt.savefig('test_2D.pdf')







plt.switch_backend('agg')
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
pa = np.array([i for i in range(p0,p1)])
X = np.array([p[prs] for prs in pa])
Y = lat
print('lat ',len(lat),'X',len(X),'circ',cir[0,:,:].shape)
X, Y = np.meshgrid(X, Y)
# Plot the surface.
surf = ax.plot_surface(Y, X, np.transpose(cir[0,:,:]), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1500, 1500) # limit values of circulation
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('test_3D.pdf')
#plt.show()



plt.switch_backend('agg')
plt.figure()
labels = []
#x_values = np.array([nb_of_bins])
#x_values = np.fromfunction(lambda i: mini + i*bin_width , (nb_of_bins,))
            #    plt.plot(x_values[:],kernel_3[:,j],linestyle = '--',label = ('model %2.0f' % self.models[j]))
x_values = lat[:] # the lat values in degrees
pre = 0
print('pressure ',pa[pr])
#for tim in range(31): # len(time)
for pr in range(len(pa)): # indices of pressure. 24 should not be in the list
     #for l in range(len(lat_grids)): # l should be replaced, in v, by lat_grids[l]
         #print('time',tim)
         plt.plot(x_values[:],cir[0,pr,:],label = ('p = %2.0f' % pa[pr])) # ,linewidth=1.0

#plt.plot(x_values[:],kernel_8[:]/np.sum(kernel_8[:]),'g-',linewidth=2.0,label = ('Total change'))
#plt.plot(x_values[:],kernel_9[:]/np.sum(kernel_9[:]),'b-',linewidth=2.0,label = ('Thermo change'))
#plt.plot(x_values[:],kernel_10[:]/np.sum(kernel_10[:]),'r-',linewidth=2.0,label = ('Kalman estimate'))
plt.title('Circulation on 01.01.1979') # at t = %2.0f' %0) # pa[pr]
plt.legend()
#plt.xlim(mini,(mini + 100*bin_width))
plt.xlabel('Latitudes (degrees)')
plt.ylabel('circulation ()')
plt.legend(loc='best')
plt.savefig('test.pdf')

















exit()
# ----------------------------------------------------------------------------------------------------------------------
# PLOT Kalman T change estimate HISTOGRAM #
# ----------------------------------------------------------------------------------------------------------------------

nb_of_bins = 100
kernel_8 = np.zeros(nb_of_bins)
kernel_9 = np.zeros(nb_of_bins)
kernel_10 = np.zeros(nb_of_bins)
kernel_11 = np.zeros(nb_of_bins)
kernel_12 = np.zeros(nb_of_bins)
kernel_13 = np.zeros(nb_of_bins)
kernel_1 = np.zeros((nb_of_bins, len(self.models)))
kernel_2 = np.zeros((nb_of_bins, len(self.models)))
kernel_3 = np.zeros((nb_of_bins, len(self.models)))
kernel_4 = np.zeros((nb_of_bins, len(self.models)))
kernel_5 = np.zeros((nb_of_bins, len(self.models)))
kernel_6 = np.zeros((nb_of_bins, len(self.models)))
kernel_h_3 = np.zeros((nb_of_bins, len(self.models)))
kernel_h_4 = np.zeros((nb_of_bins, len(self.models)))
kernel_h_5 = np.zeros((nb_of_bins, len(self.models)))
kernel_h_6 = np.zeros((nb_of_bins, len(self.models)))
# relative change
# maxi = 30                                                                                                                                                                  #mini = -15
# abs change
maxi = 0.75
mini = -0.4
bin = (maxi - mini) / float(
    nb_of_bins)  # bin length
bin_width = bin * 1.001  # if last point is equal to max, won't get error
x_grid = np.linspace(mini, maxi,
                     nb_of_bins)  # creates a 1-dimensional array of 100 values between minim and maxim
# x_values = np.fromfunction(lambda i: mini + i*bin_width , (nb_of_bins,))
x_values = np.array([nb_of_bins])
x_values = np.fromfunction(lambda i: mini + i * bin_width, (nb_of_bins,))
for j in range(len(self.models)):
    hist_1 = stats.gaussian_kde(estimate_sum[j, :], bw_method=None)
    hist_3 = stats.gaussian_kde(Tot_sum[j, :], bw_method=None)
    hist_5 = stats.gaussian_kde(Th_sum[j, :], bw_method=None)
    kernel_1[:, j] = hist_1.evaluate(x_grid)
    kernel_3[:, j] = hist_3.evaluate(x_grid)
    kernel_5[:, j] = hist_5.evaluate(x_grid)
for k in range(2):
    plt.switch_backend('agg')
    plt.figure()
    plt.title('Absolute Pr change estimate per CMIP5 model')
    labels = []
    # abs change
    fig, ax = plt.subplots(3, 3, figsize=(9, 6),
                           subplot_kw={'xticks': [-0.2, 0.0, 0.2, 0.4, 0.6], 'yticks': [0, 2, 4, 6]})
    # rel change
    # fig, ax = plt.subplots(3,3,figsize=(9, 6),subplot_kw={'xticks':[-10,0,10,20] , 'yticks': [0.0,0.04,0.08,0.12]})
    i = 0
    for r in range(3):
        for c in range(3):
            m = r * 3 + c + k * 9
            # absolute change
            # ax[r,c].plot(x_values[:],kernel_3[:,m],'g-',linewidth=1.8,label = ('Total change'))
            # ax[r,c].plot(x_values[:],kernel_5[:,m],'b-',linewidth=1.8,label = ('Thermo change'))
            # ax[r,c].plot(x_values[:],kernel_1[:,m],'r-',linewidth=1.8,label = ('Kalman estimate'))
            # ax[r,c].set_xlim(mini,mini + 100*bin_width)
            plt.xlabel('dpr/dT (mm/d/deg C)')
            # relative change
            ax[r, c].plot(x_values[:], kernel_3[:, m], 'g-', linewidth=1.8, label=('Total change'))
            ax[r, c].plot(x_values[:], kernel_5[:, m], 'b-', linewidth=1.8, label=('Thermo change'))
            ax[r, c].plot(x_values[:], kernel_1[:, m], 'r-', linewidth=1.8, label=('Kalman estimate'))
            ax[r, c].set_xlim(mini, (mini + 100 * bin_width))
            # plt.xlabel('dpr/E[pr](1970-2000)/dT  ( in %/deg C)')
            ax[r, c].set_title('model %d' % (self.models[m]))
            # ax[r, c].set_ylim(0,0.15)
            ax[r, c].set_ylim(0, 7)
            ax[r, c].set_xlim(-0.2, 0.6)
            # plt.legend(loc='best')
            fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
            # plt.legend(loc='upper right', bbox_to_anchor=(4.0, 3.0))
    plt.savefig('All_pr_Kalman_vs_raw_change_histo' + str(k) + '' + str(self.s) + '.pdf')

f.close()
exit()
#a = np.array([self.pr['timespan_'+str(j)] for j in range(1,nb_of_files_prec[0,self.m]+1)])
#pr1 = MFDataset(a)
#pr = pr1.variables['pr'][:]

#lons = pr1.variables['lon'][:] #AttributeError: 'numpy.ndarray' object has no attribute 'variables'
#time = pr1.variables['time'][:]
#lats = pr1.variables['lat'][:]

#b = np.array([self.pr_fut['timespan_'+str(j)] for j in range(1,nb_of_files_prec[1,self.m]+1)])
#pr_fut1 = MFDataset(b)
#pr_fut = pr_fut1.variables['pr'][:]
#pr1.close()




#client = boto3.client('s3') #low-level functional API

#resource = boto3.resource('s3') #high-level object-oriented API

#my_bucket = resource.Bucket('my-bucket') #subsitute this for your s3 bucket name.


