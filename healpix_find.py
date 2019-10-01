#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import healpy as hp
from astropy import utils, io
from astropy.io import fits
from astropy.table import Table, vstack, join, Column
import math as mat
import itertools as it

#--------------------------------------
# Datalab and related imports
#--------------------------------------
from dl import authClient as ac, queryClient as qc

#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------
if __name__ == "__main__":

#spits out list of "good" healpix and their RA and Dec centers
    file='nsc_instcal_combine.fits'    #file with NSC healpix 
    # PIX column is healpix number, NEXPOSURES is number of exposures over that particular healpix 
    hdul=fits.open(file) #open file
    tabl=hdul[1].data #read BINARY_TBL

    nex=3 #I want healpix with more exposures than nex
    good=tabl['NEXPOSURES']>nex #determine which healpix have nexposure>nex, which is the minimum 
                                #number of exposures I want for an area of the sky to be searched
    lst_good=tabl[good] #keeps in table only pix with enough exposures

    #convert ra and dec of good pix centers to strings so I can put them into NSC queries------------------------------------- 
    str_ra=[]
    for i in range(len(lst_good)):
        str_ra.append(lst_good["RA"][i])

    str_dec=[]
    for i in range(len(lst_good)):
        str_dec.append(lst_good["DEC"][i])

    #lim=10 # **For testing purposes  
    #find healpix with at least 3 exposures taken 20 minutes apart------------------------------------------------------------    
    good_pix=[] #array of booleans, True if good healpix
    for p in range(0,len(str_dec)): #for each healpix with sufficient exposures.  **Use to search over whole sky
    #for p in range(0,lim): # **Use for testing
        good_day=[]
    # Radial query centered on healpix center, search radius 0.5 deg (approx. radius of healpix)------------------------------
        res=qc.query(sql="".join(["select * from nsc_dr1.chip where q3c_radial_query(ra,dec,",str(str_ra[p]),",",str(str_dec[p]),",.5)"]),fmt='table')
        exp=np.unique(res["exposure"],return_index=True) #get unique exposures and indices
        exp[1].sort() #return list of unique exp. indices in order
        e_labs=np.zeros(len(res),dtype=bool) #create array of Falses, length of queried table
        e_labs[exp[1]]=True #e_labs = True if the exposure with the same index is unique
        u_res=res[e_labs] #table with unique exposures------------------------------------------------------------------------
    ##duplicate mjd column of res so I can do things but preserve the originals
        mjd_shift=Column(np.array(u_res['mjd']))  #create the column
        u_res.add_column(mjd_shift,name="mjd_shift") #add column to res
    #time shift (account for mjd starting at midnight (0.5 mjd = 1/2 day shift),and time zones(x/24 shift))-------------------
        ctio=u_res["instrument"]=='c4d'  #in chile
        ariz=u_res["instrument"]!='c4d'  #in arizona
        u_res["mjd_shift"][ctio]=u_res["mjd"][ctio]-((4.0/24)+0.5) #add 0.5 and 4/24 to mjd_correct in chile
        u_res["mjd_shift"][ariz]=u_res["mjd"][ariz]-((7.0/24)+0.5) #add 0.5 and 7/24 to mjd_correct in arizona
    #add column of shifted mjd rounded down to date (without time)------------------------------------------------------------ 
        mjd_day=[mat.floor(i) for i in u_res['mjd_shift']] #round all mjds down to nearest int
        u_res.add_column(Column(mjd_day),name='mjd_day') #add column of dates to u_res
    #loop over unique days to find those with at least 3 exposures taken 20 min. after one another---------------------------- 
        days=np.unique(u_res['mjd_day'],return_index=True)
        for ds in range(0,len(days[0])):
            day=u_res['mjd_day']==days[0][ds] #to select rows pertaining to day ds
            day_exps=u_res[day]  #to get exposures taken on day ds
            times=np.array(day_exps['mjd_shift'])  #The times of each exposure for day ds
            times.sort() #sort them
            differences=np.diff(times) #find differences between consecutive times 
            short=differences<(20.0/60/24) #True if difference is less than 20 minutes
            if len(short)>0: 
                consec=[len(list(g[1])) for g in it.groupby(short) if g[0]==True] #find instances of consecutive "True"s
                if len(consec)>0: #if there are actually consecutive exposures taken 20 minutes apart,
                    consec_max=max(consec) #the maximum number of consecutive exposures 20 minutes apart - 1
                    if consec_max>2: #If at least 3 consec. exposures,
                        good_day.append(days[0][ds]) #add day ds to the list of good days for this healpix p
        if len(good_day)!=0:  #if there are good days (with more than 3 exposures spaced 20 minutes apart) 
            good_pix.append(True) #The healpix is good
        else: good_pix.append(False)
            
    healpix_good=Column(lst_good[0:lim]['PIX'][good_pix]) #list of good healpix      
    #update lists of healpix ra and dec to only include ones of good healpix
    #ra_s=np.array(str_ra[0:lim]) # **Use for testing purposes
    ra_s=np.array(str_ra)       # **Use for whole sky
    ra_good=ra_s[good_pix]
    #dec_s=np.array(str_dec[0:lim])  # **Use for testing purposes
    dec_s=np.array(str_dec)        # **Use for whole sky
    dec_good=dec_s[good_pix]

    t=Table()
    t['healpix']=healpix_good
    #healpix_good.write("healpix_good.fits",format="fits")
    t.write("healpix_good.fits",format="fits")
    hdul.close()