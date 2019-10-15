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

from numpy import arange,array,ones,linalg
import math as m
from statistics import mean
from scipy.stats import pearsonr

import sys

#--------------------------------------
# For DBSCAN
#--------------------------------------
from sklearn.cluster import DBSCAN

#--------------------------------------
# For Silhouette
#--------------------------------------
from sklearn.metrics import silhouette_samples, silhouette_score

#--------------------------------------
# For RANSAC
#--------------------------------------
from sklearn import linear_model, datasets

#--------------------------------------
# Datalab and related imports
#--------------------------------------
from dl import authClient as ac, queryClient as qc

#-----------------------------------------------------------------------------
# Functions 
#-----------------------------------------------------------------------------

#--------------------------------------
# Remove a cluster if invalid
#--------------------------------------
def removal(cluster,out_table):
    for kt in range(0,len(cluster)): #for each point in the cluster
        clonck=out_table['measid']==cluster['measid'][kt] #using measid, find the same point in out_table
        it=clonck.tolist().index(True) #index of bad point
        out_table[it]['cluster_label']=-1 #give outlier label

#--------------------------------------
# Give points a track  label 
#--------------------------------------
def labeling(label,out_table,members,pos): #label = some number for the track label, members = track_members function outpit, pos = "p" for pred_pos, "h" for hyp_pos
    for kt in range(0,len(members)): #for each point
        clonck=out_table['measid']==members[kt] #find matching point in out_table
        it=clonck.tolist().index(True) #index of bad point
        if pos=='h': #if hyp_pos was used to predict position,
            out_table[it]['track_h']=label #give appropriate track label 
        if pos=='p':  #if pred_pos was used to predict position,  
            out_table[it]['track_p']=label #give appropriate track label

#--------------------------------------
# Spacial silhouette analysis on each cluster
#--------------------------------------
def sil(X,db,out_table,unique_labels,min_score):  #Find points with low silhouette scores, remove them!
    samps=silhouette_samples(X,db.labels_) #silhouette score for each clustered point
    for i in range(0,max(unique_labels)+1): #for each cluster, except the outliers
        cluster=out_table[db.labels_==i] #define the cluster
        cluster_samps=samps[db.labels_==i] #get the cluster's samps
        for j in cluster_samps: #loop through every point j in cluster i, and its score 
            if j<min_score: #if the score is below min_score(0.85 seems good, so does 0.8. Investigate)
                bad=out_table['measid']==cluster['measid'][cluster_samps==j] #bad points matched by measid
                ind=bad.tolist().index(True) #index of bad points
                out_table[ind]['cluster_label']=-1 #set cluster label of bad points to -1 (cluster label of outliers)

#--------------------------------------
# Calculate pearson correlation coefficient of cluster 
#--------------------------------------
def peacc(Cluster,spacetime):
    x=Cluster['ra'] #RA coord.s of cluster members (X)
    if spacetime=='s': #if spacial,
        y=Cluster['dec'] #Dec coord.s of cluster members  (Y_space)
    if spacetime=='t': #if temporal,
        y=Cluster['mjd'] #time of measurement of cluster members (Y_time)
    pearson=pearsonr(x,y)[0] #calculate PCC of cluster 
    return(pearson)     

#--------------------------------------
# RANSAC analysis to remove linearity outliers of cluster in ra-mjd space, 
# also calculates velocity of cluster (at this point, a tracklet) 
#--------------------------------------
def ranslap(cluster,out_table):     
    ra=np.array(cluster['ra']) #RA coord.s of cluster members 
    dec=np.array(cluster['dec']) #Dec coords of cluster members
    Xra=np.reshape(ra,(len(ra),1)) #reshape RA coord.s array (X_ra)
    Xdec=np.reshape(dec,(len(dec),1)) #reshape Dec coord.s array  (X_dec)
    y=np.array(cluster['mjd']) #time of measurement of cluster members (Y)

    if len(np.unique(y))<3: #if there are fewer than 3 unique measurement times (fewer than 3 moving object detections)
        inlier_mask=np.zeros(len(ra), dtype=bool) #list of zeros
        outlier_mask=np.ones(len(ra), dtype=bool) #list of ones
        clu=cluster[inlier_mask] #cluster inliers 
        no=cluster[outlier_mask] #cluster outliers
        return(clu,no,0,0) #does not RANSAC but gives all members "outlier" labels
    else: #if there are at least 3 independent moving object detections in the cluster,
        # Robustly fit linear model with RANSAC algorithm-----------------------------------------
        ransac_ra = linear_model.RANSACRegressor(residual_threshold=.001)
        ransac_ra.fit(Xra, y) #RANSAC on RA and MJD
        
        ransac_dec = linear_model.RANSACRegressor(residual_threshold=.001)
        ransac_dec.fit(Xdec, y) #RANSAC on Dec and MJD

        # Predict data of estimated models--------------------------------------------------------
        line_Xra = np.reshape(np.arange(Xra.min(), Xra.max()+.0001,step=((Xra.max()+.0001)-Xra.min())/20),(20,1))
        line_y_ransac_ra = ransac_ra.predict(line_Xra) #line for RANSAC fit, ra
        
        line_Xdec = np.reshape(np.arange(Xdec.min(), Xdec.max()+.0001,step=((Xdec.max()+.0001)-Xdec.min())/20),(20,1))
        line_y_ransac_dec = ransac_dec.predict(line_Xdec) #line for RANSAC fit, dec
    
        xsra=np.concatenate(line_Xra) #x values of RA,MJD RANSAC line
        xsdec=np.concatenate(line_Xdec) #x values of Dec,MJD RANSAC line
        ysra=line_y_ransac_ra #y values of RA,MJD RANSAC line
        ysdec=line_y_ransac_dec #y values of Dec,MJD RANSAC line
        
        mra = (ysra[-1]-ysra[0])/(xsra[-1]-xsra[0]) # 1/slope of RA,MJD RANSAC (1/velocity in RA)
        mdec = (ysdec[-1]-ysdec[0])/(xsdec[-1]-xsdec[0]) # 1/slope of Dec,MJD RANSAC (1/velocity in Dec)
        
        inlier_mask = ransac_ra.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        clu=cluster[inlier_mask] #cluster after outlier removal (cluster inliers)
        no=cluster[outlier_mask] #cluster outliers
        
        if mra!=0 and mdec!=0: #if the slopes are not zero,
            return (clu,no,1/mra,1/mdec)  #This is the velocity of the object that the tracklet (cluster) represents in ra & dec directions
        else: #if the slopes ARE zero,
            return(clu,no,0,0) #return "0" as velocities 

#--------------------------------------
# Validate each cluster using functions "peacc" and "ranslap"
#--------------------------------------
def validate_it(X,db,out_table,labels):  #Determine invalid points in each cluster and remove, via RANSAC 
    for i in range(0,max(labels)+1): #for each cluster i, except the outliers
        clust=out_table['cluster_label']==i #define the cluster
        cluster=out_table[clust]

        if len(cluster)>1:   #if there's more than 1 member in the cluster,
            #Space
            pp=peacc(cluster,'s') #calculate PCC of cluster, spacially
            if abs(pp)<0.9: #if spacial PCC is too low, get rid of cluster!  This is after silhouette, so I think it'll be fine?
                removal(cluster,out_table) #gets rid of cluster (gives points "outlier" label)
            else: #if PCC is high enough,
            #Time
                time=ranslap(cluster,out_table) #ransac on the cluster
                if len(time[1])>0: #If there are outliers,
                    removal(time[1],out_table)  #give the outliers "outlier" label       
                
                now=out_table['cluster_label']==i #after RANSAC on cluster,
                now_cluster=out_table[now] #define the new cluster, sans outliers
                if len(np.unique(now_cluster['mjd']))>2: #if the cluster has more than 2 measurements,
                    pt=peacc(now_cluster,'t') #find PCC, temporally 
                    if abs(pt)<0.9: #if temporal PCC is too low, get rid of cluster!
                        removal(now_cluster,out_table) #give cluster points "outlier" labels
                else: #if cluster has fewer than 3 measurements,
                    removal(now_cluster,out_table) #get rid of cluster!
        #Time to figure out if any of the new clusters have fewer than 3 points, because I don't want those!
        #Tracklets need to have 3 or more measurements! If so, gve them their appropriate velocities.
        new=out_table['cluster_label']==i
        new_cluster=out_table[new] #new cluster
        if len(new_cluster)<3 or (time[2]==0 and time[3]==0): #if the length of the cluster is less than 3, or tracklet velocity=0, make 'em all outliers!
            removal(new_cluster,out_table) #give cluster "outlier" label
        else: #if the length of the cluster is greater than 3, give them the appropriate velocities 
            v_ra=time[2] #define tracklet velocity in RA
            v_dec=time[3] #define tracklet velocity in dec
            for plu in range(0,len(out_table)): #for every measurement
                if out_table['cluster_label'][plu]==i: #if the measurement corresponds to cluster i,
                    out_table['v_ra'][plu]=v_ra #give it the tracklet velocity in RA
                    out_table['v_dec'][plu]=v_dec #and give it the tracklet velocity in De 

#--------------------------------------
# Predict tracklet member positions using corresponding tracklet velocities
#--------------------------------------
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
        if p['cluster_label']!=-1: #if cluster label is NOT outlier (-1),
            ra_pred=p['v_ra']*(too-p['mjd'])+p['ra'] #calculate predicted RA
            dec_pred=p['v_dec']*(too-p['mjd'])+p['dec'] #calculate predicted DEC
            dev=abs(pow((pow(p['v_ra'],2)+pow(p['v_dec'],2)),0.5)*7) #distance a point in a cluster (tracklet) could have traveled in 7 days 
            devv=abs(pow(pow((ra_pred-p['ra']),2)+pow((dec_pred-p['dec']),2),0.5)) #distance actually traveled
            if devv<dev: #if the distance traveled is less than the distance the tracklet COULD have traveled in 7 days,
                mid.append(p['measid'])  #add the point's measurement id to mid
                lab.append(p['cluster_label']) #add cluster label to lab
                ra_ps.append(ra_pred) #add predicted RA to ra_ps
                dec_ps.append(dec_pred) #add predicted Dec to dec_ps
    return(mid,ra_ps,dec_ps,lab) #returns measurement id's, predicted RA & Dec, and cluster label

#--------------------------------------
# Predict hypothetical point positions using one tracklet's velocity 
#--------------------------------------
def hyp_pos(table,cluster,t=[],to=[]): #table = out-table, t = time of desired pred_pos.(see below for specifications), cluster = cluster label (#)
    mid=[] #empty array for measurement id 
    lab=[] #empty array for cluster label
    ra_ps=[] #empty array for predicted RA
    dec_ps=[] #empty array for predicted Dec
    clustie=table['cluster_label']==cluster 
    tab=table[clustie] #define the cluster whose velocities you're using 
    vevra=tab['v_ra'][0] #cluster velocity in RA
    vevdec=tab['v_dec'][0] #cluster velocity in Dec
    dev=abs(pow((pow(vevra,2)+pow(vevdec,2)),0.5)*7) #distance a point could have traveled in 7 days at cluster (tracklet) velocity 
    for p in table: #for every entry (measurement, point) in out-table
        if t==0:  #do this if you just want to advance all the times in out-table by amount "to"
            too=p['mjd']+to #in this case you'd set t=0, and to=however much time you want to advance by
        else: #this is when you want to see predicted positions at some specific mjd "t", and set to=0
            too=t
        ra_pred=vevra*(too-p['mjd'])+p['ra'] #calculate predicted RA
        dec_pred=vevdec*(too-p['mjd'])+p['dec'] #calculate predicted DEC
        devv=abs(pow(pow((ra_pred-p['ra']),2)+pow((dec_pred-p['dec']),2),0.5)) #distance actually traveled
        if devv<dev: #if the distance traveled is less than the point COULD have traveled at cluster's velocity in 7 days, 
            ra_ps.append(ra_pred) #add predicted RA to ra_ps
            dec_ps.append(dec_pred) #add predicted Dec to dec_ps
            mid.append(p['measid'])  #add the measurement id to mid
            lab.append(p['cluster_label']) #add cluster label to lab
    return(mid,ra_ps,dec_ps,lab) #returns measurement id's, hypothetical RA & Dec, and cluster label

#--------------------------------------
# Find new track members after projecting measurement positions to common time using pred_pos or hyp_pos
#--------------------------------------
def track_members(cluster,pos,out_table): #cluster = the cluster whose velocity you used to project point positions. pos = the hyp_pos or pred_pos you ran using cluster's velocity
    #cl_ra=[] #empty array for measurement RA
    #cl_dec=[] #empty array for measurement Dec
    cl_id=[] #empty array for measurement id
    clo=out_table['cluster_label']==cluster 
    cc=out_table[clo] #define the cluster whose velocity you used to project point positions 
    center_ra=cc['ra'][0] #RA of the cluster at the time you projected the positions to 
    center_dec=cc['dec'][0] #Dec of the cluster at the time you projected the positions to 
    for i in range(0,len(pos[1])): #for every point projected,
        position=pow(pow((pos[1][i]-center_ra),2)+pow((pos[2][i]-center_dec),2),0.5) #find the distance from RA,Dec of cluster
        if position<0.005: #if that distance is less than my decided max distance,
            #cl_ra.append(hypo_pos[1][i])
            #cl_dec.append(hypo_pos[2][i])
            cl_id.append(pos[0][i]) #append the point's measurement id to the output list
    return(cl_id) #returns track members' measurement id's

#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    pix = sys.argv[1] #healpix number
    mark = sys.argv[2] #1 = adequate exposure spacing (run full analysis), 0 = inadequate (only remove SOs) 
    #RA=201.34
    #DEC=-15.74
    #healpix=hp.ang2pix(128,RA,DEC,lonlat=True) #get the healpix number
    RA=hp.pix2ang(128,int(pix),lonlat=True)[0]
    DEC=hp.pix2ang(128,int(pix),lonlat=True)[1]
    nbs=hp.get_all_neighbours(512,RA,DEC,lonlat=True) #get the 8 nearest neighbors to the cooordinates for nside=512
    coords=hp.pix2ang(512,nbs,lonlat=True) #get the center coordinates for the8 nside=512  neighbors
    fpix=np.unique(hp.ang2pix(256,coords[0],coords[1],lonlat=True)) #find the 4 unique corresponding nside=256 helapix 
    dat=qc.query(sql="".join(["SELECT meas.mjd,meas.ra,meas.dec,meas.measid FROM nsc_dr1.meas as meas JOIN nsc_dr1.object as obj on objectid=obj.id WHERE obj.ring256=",str(fpix[0])," or obj.ring256=",str(fpix[1])," or obj.ring256=",str(fpix[2])," or obj.ring256=",str(fpix[3])]),fmt='table')

    X=np.column_stack((np.array(dat['ra']),np.array(dat['dec'])))  #coordinates of measurements---------------------------

    if len(dat)>0: #if there is actually any data,
#-------------------
# SO Identification
#-------------------
    #Compute DBSCAN on all measurements ----------------------------------------------------------------------------------
        db_1 = DBSCAN(eps=0.000138889, min_samples=2).fit(X) #eps=0.000139=0.5" (same spacing used to create NSC object table), to cluster SOs
    #Get outliers from db_1 ----------------------------------------------------------------------------------------------
        outliers=db_1.labels_==-1 #define outliers from first DBSCAN labels
        X_out=np.column_stack((np.array(dat['ra'][outliers]),np.array(dat['dec'][outliers]))) #coordinates of DBSCAN outliers
    #Compute DBSCAN on outliers ------------------------------------------------------------------------------------------
#--------------------
# Tracklet Formation 
#--------------------
        db_2 = DBSCAN(eps=.003, min_samples=3).fit(X_out) #min_samples=3 so at least 3 mmts in a tracklet, to cluster FMOs
    #Analyze results of db_2 ---------------------------------------------------------------------------------------------
        t_out=dat[outliers]#Create table of outliers from db_1 (out_table)
        t_out.add_column(Column(db_2.labels_),name="cluster_label") #Add their cluster labels from db_2
        lab=np.unique(db_2.labels_) #Identify the unique cluster labels from db_2
        if len(lab)>1: #if more than just outliers,
            t_out.add_column(Column(np.zeros(len(t_out),dtype=float)),name="v_ra") #add RA velocity column to out_table
            t_out.add_column(Column(np.zeros(len(t_out),dtype=float)),name="v_dec") #add Dec velocity column to out_table
#---------------------
# Tracklet Validation
#---------------------           
            sil(X_out,db_2,t_out,lab,0.8) #Silhouette analyze it!
            validate_it(X_out,db_2,t_out,lab) #PCC and RANSAC it!  
            if len(np.unique(t_out['cluster_label']))-1>0: #if there are still clusters,
                t_out.add_column(Column(np.repeat(pix,len(t_out))),name='pix') #give them a healpix label
                t_out.add_column(Column(np.zeros(len(t_out))),name='track_p') #give it a track_p numbering column
                t_out.add_column(Column(np.zeros(len(t_out))),name='track_h') #give it a track_h numbering column
#-----------------
# Track Formation
#-----------------                
                n=0 #for track labels
                for cl in np.unique(t_out['cluster_label']): #for every unique validated cluster,
                    clu=t_out['cluster_label']==cl
                    clus=t_out[clu] #define the cluster
                    my_time=clus['mjd'][0] #define the first measurement of the cluster (time to project points to)
                    pos=pred_pos(t_out,t=my_time) #predict all tracklet positions using their unique velocities
                    hyp=hyp_pos(t_out,cl,t=my_time) #hypothesize all point positions using cluster's velocity 
                    p_mems=track_members(cl,pos,t_out) #track members under use of pred_pos
                    h_mems=track_members(cl,hyp,t_out) #track members under use of hyp_pos
                    cluster_length=len(clus['measid']) #number of original cluster members
                    pluster_length=len(p_mems) #number of predicted track members
                    hluster_length=len(h_mems) #number of hypothesized track members
                    if pluster_length>cluster_length: #if there are any new predicted track members,
                        n=n+1 #track label
                        labeling(n,t_out,p_mems,'p')#put track label on t_out rows of all points in track (track_p)
                        if hluster_length>cluster_length: #if there are any new hypothesized track members,
                            labeling(n,t_out,h_mems,'h')#put track marker on t_out rows of all points in track (track_h)
                    else: 
                        n=n        
        t_out.write("../../../mnt/lustrefs/scratch/katie.fasbender/healpix_%d.fits" % int(pix),format="fits",overwrite=True) #write a fits file with all measurements not associated with SOs, and their info
        #t_out.write("healpix_%d.fits" % int(pix),format="fits",overwrite=True)
        #t.write("../../../mnt/lustrefs/scratch/katie.fasbender/healpix_good.fits",format="fits",overwrite=True)


        