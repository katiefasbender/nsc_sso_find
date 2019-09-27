#!/usr/bin/env python

#--------------------------------------
# imports
#--------------------------------------
import numpy as np
import healpy as hp
#from __future__ import print_function #to use print() as a func. in py2
from astropy import utils, io
from astropy.io import fits
from astropy.table import Table, vstack, join, Column
import math as mat
#import fitsio
import itertools as it

from numpy import arange,array,ones,linalg
import math as m
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from statistics import mean

from scipy.stats import pearsonr

#%matplotlib inline
#%matplotlib notebook

#--------------------------------------
# For DBSCAN
#--------------------------------------
from sklearn.cluster import DBSCAN

#--------------------------------------
# For Silhouette
#--------------------------------------
#from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

#--------------------------------------
# For RANSAC
#--------------------------------------
from sklearn import linear_model, datasets

#--------------------------------------
# Datalab and related imports
#--------------------------------------
# need these for authenticating and for issuing database queries
from dl import authClient as ac, queryClient as qc
# need storeClient to use virtual storage or myDB
# Get helpers for various convenience function
from dl import storeClient as sc
from dl.helpers.utils import convert
from getpass import getpass

def removal(cluster,out_table):
    for kt in range(0,len(cluster)):
                            clonck=out_table['measid']==cluster['measid'][kt]
                            it=clonck.tolist().index(True) #index of bad points
                            out_table[it]['cluster_label']=-1 

def labeling(label,out_table,members,pos):
    ids=members[0]
    for kt in range(0,len(ids)):
                            clonck=out_table['measid']==ids[kt]
                            it=clonck.tolist().index(True) #index of bad points
                            if pos=='h':
                                out_table[it]['track_h']=label
                            if pos=='p':    
                                out_table[it]['track_p']=label

def sil(X,db,out_table,unique_labels,min_score):  #Find points with low silhouette scores, remove them!
    samps=silhouette_samples(X,db.labels_) #silhouette score for each point
    for i in range(0,max(unique_labels)+1): #for each cluster, except the outliers
        cluster=out_table[db.labels_==i] #define the cluster
        cluster_samps=samps[db.labels_==i] #get the cluster's samps
        for j in cluster_samps: #loop through every point j in cluster i, and its score 
            if j<min_score: #if the score is below min_score(0.85 seems good, so does 0.8. Investigate)
                bad=out_table['measid']==cluster['measid'][cluster_samps==j] #bad points = ones whose object id 
                                                            #matches the object id of the cluster point with score j
                ind=bad.tolist().index(True) #index of bad points
                out_table[ind]['cluster_label']=-1 #set cluster label of bad points to -1 (cluster label of outliers)

def peacc(Cluster,spacetime):
    x=Cluster['ra']
    if spacetime=='s':
        y=Cluster['dec']
    if spacetime=='t':
        y=Cluster['mjd']
    pearson=pearsonr(x,y)[0]
    return(pearson)     

def ranslap(cluster,out_table):     
   
    ra=np.array(cluster['ra'])
    dec=np.array(cluster['dec'])
    Xra=np.reshape(ra,(len(ra),1))
    Xdec=np.reshape(dec,(len(dec),1))
    y=np.array(cluster['mjd'])

    #if max(y)-min(y)==0:
   # print(np.unique(y))
    if len(np.unique(y))<3:
        inlier_mask=np.zeros(len(ra), dtype=bool)
        #print("inlier mask = ",inlier_mask)
        outlier_mask=np.ones(len(ra), dtype=bool)
        clu=cluster[inlier_mask] #cluster inliers
        no=cluster[outlier_mask] #cluster outliers
        return(clu,no,0,0) #keeps cluster as is, does not ransac it but gives all members outlier labels
    else:
# Robustly fit linear model with RANSAC algorithm-----------------------------------------
        ransac_ra = linear_model.RANSACRegressor(residual_threshold=.001)
        ransac_ra.fit(Xra, y)
        
        ransac_dec = linear_model.RANSACRegressor(residual_threshold=.001)
        ransac_dec.fit(Xdec, y)

# Predict data of estimated models--------------------------------------------------------
        line_Xra = np.reshape(np.arange(Xra.min(), Xra.max()+.0001,step=((Xra.max()+.0001)-Xra.min())/20),(20,1))
        line_y_ransac_ra = ransac_ra.predict(line_Xra) #line for RANSAC fit, ra
        
        line_Xdec = np.reshape(np.arange(Xdec.min(), Xdec.max()+.0001,step=((Xdec.max()+.0001)-Xdec.min())/20),(20,1))
        line_y_ransac_dec = ransac_dec.predict(line_Xdec) #line for RANSAC fit, dec
    
        xsra=np.concatenate(line_Xra)
        xsdec=np.concatenate(line_Xdec)
        ysra=line_y_ransac_ra
        ysdec=line_y_ransac_dec
        
        mra = (ysra[-1]-ysra[0])/(xsra[-1]-xsra[0])
        mdec = (ysdec[-1]-ysdec[0])/(xsdec[-1]-xsdec[0])
        
        inlier_mask = ransac_ra.inlier_mask_
        #print("inlier mask = ",inlier_mask)
        outlier_mask = np.logical_not(inlier_mask)
        clu=cluster[inlier_mask] #cluster after outlier removal (cluster inliers)
        no=cluster[outlier_mask] #cluster outliers
        
        if mra!=0 and mdec!=0:
            return (clu,no,1/mra,1/mdec)  #This is the velocity of the object that the tracklet (cluster) represents in ra & dec directions
        else:
            return(clu,no,0,0)

def validate_it(X,db,out_table,labels):  #Determine invalid points in each cluster and remove, via RANSAC 
    for i in range(0,max(labels)+1): #for each cluster, except the outliers
        clust=out_table['cluster_label']==i #define the cluster
        cluster=out_table[clust]

        if len(cluster)>1:   
        #Space
            pp=peacc(cluster,'s')
            #print("pp = ",pp,", cluster = ",i)
            if abs(pp)<0.9: #if the pcc is too low, get rid of cluster!  This is after silhouette, so I think it'll be fine?
                #get indices of cluster 
                removal(cluster,out_table)
            else:
        #Time
                time=ranslap(cluster,out_table) #ransac on the cluster
                if len(time[1])>0: #If there are outliers,
                    #print("outliers for cluster ",i)
                    removal(cluster,out_table)         
                
                now=out_table['cluster_label']==i
                now_cluster=out_table[now]
                if len(np.unique(now_cluster['mjd']))>2:
                    pt=peacc(now_cluster,'t')
                    #print("pt = ",pt,", cluster = ",i)
                    if abs(pt)<0.9:
                        removal(now_cluster,out_table)
                else:
                    #print("nope!")
                    removal(now_cluster,out_table)
        #Time to figure out if any of the new clusters have fewer than 3 points, because I don't want those!
        #Tracklets need to have 3 or more measurements!
        new=out_table['cluster_label']==i
        new_cluster=out_table[new]
        
        if len(new_cluster)<3 or (time[2]==0 and time[3]==0): #if the length of the cluster is less than 3, make 'em all outliers!
            removal(new_cluster,out_table)
        else: #if the length of the cluster is greater than 3, give them the appropriate velocities 
            v_ra=time[2]
            v_dec=time[3]
            for plu in range(0,len(out_table)): #for every point p in cluster c, put cluster velocity in table
                if out_table['cluster_label'][plu]==i:
                    out_table['v_ra'][plu]=v_ra
                    out_table['v_dec'][plu]=v_dec

def pred_pos(table,t=[],to=[]): #table = out-table, t = time of desired pred_pos.(see below for specifications)
    mid=[] #empty array for measurement id 
    lab=[] #empty array for cluster label
    ra_ps=[] #empty array for predicted ra
    dec_ps=[] #empty array for predicted dec
    for p in table: #for every entry (measurement, point) in out-table
        if t==0:  #do this if you just want to advance all the times in out-table by amount "to"
            too=p['mjd']+to #in this case you'd set t=0, and to=however much time you want to advance by
        else: #this is when you want to see predicted positions at some specific mjd "t", and set to=0
            too=t
        if p['cluster_label']!=-1: #if cluster label is NOT outlier (-1)
            ra_pred=p['v_ra']*(too-p['mjd'])+p['ra'] #calculate predicted RA
            #if ra_pred>360:
            #    ra_pred=ra_pred-(360*np.floor(ra_pred/360))
            dec_pred=p['v_dec']*(too-p['mjd'])+p['dec'] #calculate predicted DEC
            #if np.abs(dec_pred)>90:
            #    dec_pred=dec_pred-(90*np.floor(dec_pred/90))
            dev=abs(pow((pow(p['v_ra'],2)+pow(p['v_dec'],2)),0.5)*7) #distance could have traveled in 7 days
            devv=abs(pow(pow((ra_pred-p['ra']),2)+pow((dec_pred-p['dec']),2),0.5)) #distance traveled
            #if devv<dev:
            mid.append(p['measid'])  #add the measurement id to mid
            lab.append(p['cluster_label']) #add cluster label to lab
            ra_ps.append(ra_pred) #add predicted RA to ra_ps
            dec_ps.append(dec_pred) #add predicted Dec to dec_ps
    return(mid,ra_ps,dec_ps,lab) #returns measurement id's, predicted RA & Dec, and cluster  label

def hyp_pos(table,cluster,t=[],to=[]): #table = out-table, t = time of desired pred_pos.(see below for specifications)
    mid=[] #empty array for measurement id 
    lab=[] #empty array for cluster label
    ra_ps=[] #empty array for predicted ra
    dec_ps=[] #empty array for predicted dec
    clustie=table['cluster_label']==cluster
    tab=table[clustie]
    vevra=tab['v_ra'][0]
    vevdec=tab['v_dec'][0]
    dev=pow((pow(vevra,2)+pow(vevdec,2)),0.5)*7
    #print("distance = ",dev)
    for p in table: #for every entry (measurement, point) in out-table
        if t==0:  #do this if you just want to advance all the times in out-table by amount "to"
            too=p['mjd']+to #in this case you'd set t=0, and to=however much time you want to advance by
        else: #this is when you want to see predicted positions at some specific mjd "t", and set to=0
            too=t
        #if p['cluster_label']!=-1: #if cluster label is NOT outlier (-1)
        ra_pred=vevra*(too-p['mjd'])+p['ra'] #calculate predicted RA
            #if ra_pred>360:
            #    ra_pred=ra_pred-(360*np.floor(ra_pred/360))
        dec_pred=vevdec*(too-p['mjd'])+p['dec'] #calculate predicted DEC
            #if np.abs(dec_pred)>90:
            #    dec_pred=dec_pred-(90*np.floor(dec_pred/90))
        #mid.append(p['measid'])  #add the measurement id to mid
        #lab.append(p['cluster_label']) #add cluster label to lab
        devv=abs(pow(pow((ra_pred-p['ra']),2)+pow((dec_pred-p['dec']),2),0.5))
        if devv<abs(dev):
            ra_ps.append(ra_pred) #add predicted RA to ra_ps
            dec_ps.append(dec_pred) #add predicted Dec to dec_ps
            mid.append(p['measid'])  #add the measurement id to mid
            lab.append(p['cluster_label']) #add cluster label to lab
        #else: 
        #    ra_ps.append(0)
        #    dec_ps.append(0)
    return(mid,ra_ps,dec_ps,lab) #returns measurement id's, predicted RA & Dec, and cluster  label

def track_members(cluster,hypo_pos,out_table): #cluster = the cluster whose velocity you used to project all the points. hypo_pos = the hyp_pos you ran using cluster's velocity
    cl_ra=[]
    cl_dec=[]
    cl_id=[]
    cl=out_table['cluster_label']==cluster
    cc=out_table[cl]
    center_ra=cc['ra'][0]
    center_dec=cc['dec'][0]
    for i in range(0,len(hypo_pos[1])):
        position=pow(pow((hypo_pos[1][i]-center_ra),2)+pow((hypo_pos[2][i]-center_dec),2),0.5)
        if position<0.005:
            cl_ra.append(hypo_pos[1][i])
            cl_dec.append(hypo_pos[2][i])
            cl_id.append(hypo_pos[0][i])
    return(cl_id,cl_ra,cl_dec)


if __name__ == "__main__":

	#pix = sys.argv[1]	

#file='C:\Users\Shady Katie\Documents\Research\dat_m.fits' 
#hdul=fits.open(file)
#dat_m=hdul[1].data
	#X=np.column_stack((dat_m['ra'],dat_m['dec']))
	#for hpix in range(0,len(ra_good)): #no for loop@ just one pix value and then you need to find the RA and DEC of that pix. 
    	#print('hpix = ',hpix)
#Define RA, DEC
	#coords=np.pix2ang(128,pix,lonlat=true)
    #RA=coords[0]
    #DEC=coords[1]
    RA=201.34
    DEC=-15.74
    nbs=hp.get_all_neighbours(512,RA,DEC,lonlat=True)
    coords=hp.pix2ang(512,nbs,lonlat=True)
    fpix=np.unique(hp.ang2pix(256,coords[0],coords[1],lonlat=True))
    dat=qc.query(token,sql="".join(["SELECT meas.mjd,meas.ra,meas.dec,meas.measid FROM nsc_dr1.meas as meas JOIN nsc_dr1.object as obj on objectid=obj.id WHERE obj.ring256=",str(fpix[0])," or obj.ring256=",str(fpix[1])," or obj.ring256=",str(fpix[2])," or obj.ring256=",str(fpix[3])]),fmt='table')

	X=np.column_stack((np.array(dat['ra']),np.array(dat['dec'])))  #coordinates of measurements---------------------------

	if len(dat)>0: #if there is actually any data......
#Compute DBSCAN on all measurements ----------------------------------------------------------------------------------
    	db_1 = DBSCAN(eps=0.000138889, min_samples=2).fit(X) #eps=0.000139=0.5" (same spacing used to create NSC object table)
#Get outliers from db_1 ----------------------------------------------------------------------------------------------
    	outliers=db_1.labels_==-1 #define outliers from first DBSCAN labels
    	X_out=np.column_stack((np.array(dat['ra'][outliers]),np.array(dat['dec'][outliers]))) #coordinates of DBSCAN outliers
#Compute DBSCAN on outliers ------------------------------------------------------------------------------------------
    	db_2 = DBSCAN(eps=.003, min_samples=3).fit(X_out) #min_samples=3 so at least 3 mmts in a tracklet
#Analyze results of db_2 ---------------------------------------------------------------------------------------------
    	t_out=dat[outliers]#Create table of outliers from db_1
    	t_out.add_column(Column(db_2.labels_),name="cluster_label") #Add their cluster labels from db_2
    	lab=np.unique(db_2.labels_) #Identify the unique cluster labels from db_2
    	if len(lab)>1: #if more than just outliers,
        	t_out.add_column(Column(np.zeros(len(t_out),dtype=float)),name="v_ra")
        	t_out.add_column(Column(np.zeros(len(t_out),dtype=float)),name="v_dec") 
        	sil(X_out,db_2,t_out,lab,0.8) #Silhouette analyze it!
            #ransac_it(X_out,db_2,t_out,lab) #RANSAC it!
        	validate_it(X_out,db_2,t_out,lab) #RANSAC it!  
        #print('# Clusters Before Validation: ',len(lab)-1)
        #print('# Clusters After Validation:',len(np.unique(t_out['cluster_label']))-1)
        	if len(np.unique(t_out['cluster_label']))-1>0: #if there are clusters,
                #print("healpix = ",healpix_good[hpix])
          		t_out.add_column(Column(np.repeat(healpix_good[hpix],len(t_out))),name='healpix') #give them a healpix label
                #print(t_out)
            	t_out.add_column(Column(np.zeros(len(t_out))),name='track_p') #give it a track_p numbering column
            	t_out.add_column(Column(np.zeros(len(t_out))),name='track_h') #give it a track_h numbering column
            	n=0
            	for cl in np.unique(t_out['cluster_label']):
               		clu=t_out['cluster_label']==cl
               		clus=t_out[clu]
               		my_time=clus['mjd'][0]
               		pos=pred_pos(t_out,t=my_time)
               		hyp=hyp_pos(t_out,cl,t=my_time)
               		clust=np.array(pos[3])==cl
               		cluster_length=len(np.array(pos[3])[clust])
               		pluster_length=len(track_members(cl,pos,t_out)[0])
               		hluster_length=len(track_members(cl,hyp,t_out)[0])
               		if pluster_length>cluster_length:
                   		n=n+1
                   		labeling(n,t_out,track_members(cl,pos,t_out),'p')#put track marker on t_out rows of all points in track (track_p)
                   		if hluster_length>cluster_length:
                       		labeling(n,t_out,track_members(cl,hyp,t_out),'h')#put track marker on t_out rows of all points in track (track_h)
               		else: 
                   		n=n
                    #print('Number of points in cluster %s: ' % cl,len(np.array(pos[3])[clust]))
                    #print("Number of points that land on cluster %s's location (p): " % cl,len(track_members(cl,pos,t_out)[0]))
                    #print("Number of points that land on cluster %s's location (h): " % cl,len(track_members(cl,hyp,t_out)[0]))
    	t_out.write("healpix_%d.fits" % healpix_good[hpix],format="fits")