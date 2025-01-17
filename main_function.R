
rm(list = ls())
require(ncdf4)
require(foreach)
require(parallel)
require(doParallel)
require(glmnet)
require(abind)
require(zoo)
gc()

####The following code is used to decompose the changes in Ccomp-related droughts
#"get_indata" is the function used to calculate the inter-annual trend in LAI, Tas, or Prec across all realizations; "fp" is the file path; "laiallna" is the file names; "coreset" is parallel number of cores; "dalen" is the total length of data; "dalab" is the matrix of "dalen"; "kmons" and "kmon" are pentad code for Southern and Northern Hemisphere, respectively; "reid_kk" is regional code. 
#"calre_laitas" is the function used to train ridge regression and decompose the changes for each dataset; "laispada", "tasspada", and "prespada" are the dataset calculated from "get_indata", resprectively;"cobsym_eo" is Ccomp-related droughts
#"pref" is the big function that used for decomposing the changes at each pentad and that includes "get_indata" and "calre_laitas"; "outna" is the name for saving results.

get_indata = function(fp,laiallna,coreset,dalen,dalab,kmons,kmon,reid_kk){
  
  chazhi = function(mat){
    na_columns = which(colSums(is.na(mat)) == nrow(mat))
    nona_columns = which(colSums(is.na(mat)) == 0)
    
    if(length(na_columns) > 0){#col=1
      for (col in na_columns) { mat[, col] <- mat[, nona_columns[which.min(abs(col - nona_columns))]]}
    }else{
      mat = mat
    }
    return(mat)
  }
  
  setwd(fp)
  cl = makeCluster(coreset)
  registerDoParallel(cl)
  laispada = foreach (k = 1:length(laiallna), .packages = c('abind','zoo')) %dopar%  { #k=1
    dat = readRDS(laiallna[k])[,,dalen]
    dat = abind(dat[,1:38,dalab[,kmons]], dat[,39:93,dalab[,kmon]], along=2)
    dat = aperm(dat, c(2,1,3))[93:1,,]
    #dim(dat)
    #dim(reid_kk)
    dat = apply(reid_kk, 1, function(pos){dat[pos[1], pos[2],]})
    dat = chazhi(dat)
    if(sum(is.na(dat)) > 0) { dat = apply(dat, 2, na.locf, na.rm = FALSE) }
    
    rclai = apply(dat, 2, function(x1){coef(lm(x1 ~ c(1:length(x1))))[2]})
    
    list(dat = dat ,
         rc = rclai)
  }
  stopCluster(cl)
  gc()
  
  return(laispada)
}
calre_laitas = function(laispada,tasspada,prespada,cobsym_eo,reid_kk){
  
  maxncol = ncol(cobsym_eo)
  modid = rep(1:maxncol,100)
  
  if(maxncol <= 13){ 
    maxtimes =  maxncol
  }else{
    maxtimes = 13  }
  
  cl = makeCluster(22)
  registerDoParallel(cl)
  reout = foreach (kloop = 1:length(modid), .packages = c('glmnet', 'zoo')) %dopar% {#kloop=5
    
    y = cobsym_eo[,modid[kloop]]
    xnum = sample(01:length(laispada))[01:maxtimes]
    trainnum = length(xnum) - 1
    
    yall     = c()
    xall_lai = list()
    xall_tas = list()
    xall_pre = list()
    for (kk in 1:length(xnum)) {#kk=11
      
      modsel = c(1:maxncol)
      modsel = modsel[-which(modsel == modid[kloop])]
      yi = cobsym_eo[, modsel[kk]]
      yall = c(yall, yi)
      
      xall_lai[[kk]] = laispada[[xnum[kk]]]$dat
      xall_tas[[kk]] = tasspada[[xnum[kk]]]$dat
      xall_pre[[kk]] = prespada[[xnum[kk]]]$dat
    }
    xall = cbind(do.call(rbind, xall_lai[1:trainnum]), 
                 do.call(rbind, xall_tas[1:trainnum]),
                 do.call(rbind, xall_pre[1:trainnum]))
    yall = yall[1:nrow(xall)]
    
    cvm = cv.glmnet(xall, yall, alpha=0, nfolds=trainnum) # 
    best_lambda = cvm$lambda.min
    
    x = cbind(laispada[[xnum[kk]]]$dat, 
              tasspada[[xnum[kk]]]$dat,
              prespada[[xnum[kk]]]$dat)
    predictions = predict(cvm, s=best_lambda, newx=x)
    
    numcol = ncol(laispada[[xnum[kk]]]$dat)
    rclai_fre = coef(cvm, s=best_lambda)[2:(numcol+1)]
    rctas_fre = coef(cvm, s=best_lambda)[(numcol+2):(numcol+numcol+1)]
    rcpre_fre = coef(cvm, s=best_lambda)[(numcol+numcol+2):(numcol+numcol+numcol+1)]
    
    laieventi = sum((laispada[[xnum[kk]]]$rc)*rclai_fre)
    taseventi = sum((tasspada[[xnum[kk]]]$rc)*rctas_fre)
    preeventi = sum((prespada[[xnum[kk]]]$rc)*rcpre_fre)
    
    list(lai_fre     = laieventi                      ,
         tas_fre     = taseventi                      ,
         pre_fre     = preeventi                      ,
         all_fre     = coef(lm(y ~ c(1:length(y))))[2],
         pre         = predictions                    ,
         obs         = y                              ,
         xnum        = xnum                           )
  }
  stopCluster(cl)
  return(reout)
}
pref = function(outna,cobsym_eo){
  
  setwd('/gpfs/share/')
  lucc = readRDS('lucc_1.5degree_vegetated_area.rds')
  regionctr = readRDS('sub_region_vegetated_area.rds')
  regionctr = t(regionctr)[93:1,]
  reid = unique(na.omit(matrix(regionctr, ncol=1)))
  
  setwd('/gpfs/lai')
  regionctr = readRDS("lai_ctr_data")[,,1]
  regionctr = regionctr + lucc - lucc
  regionctr = t(regionctr)[93:1,]
  gc()
  sum(!is.na(regionctr))
  reid_kk = which(!is.na(regionctr), arr.ind = T)
  
  fplai = '/gpfs/lai/'
  fptas = '/gpfs/tas/'
  fppre = '/gpfs/pr/'
  
  for (kmon in 19:54) {#kmon=19
    print(kmon-18)
    t1 = proc.time()
   #dalen = ((1950-1850+1)*73+1):((2022-1850+1)*73)
   #dalab = matrix(1:((2022-1951+1)*73), ncol=73, byrow = T)
    
    monn = c(19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
             35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
             51,52,53,54)
    mons = c(55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,
             71,72,01,02,03,04,05,06,07,08,09,10,11,12,13,14,
             15,16,17,18)
    kmons = mons[which(monn == kmon)]
    
    coreset=25
    ######################fp,laiallna,coreset,dalen,dalab,kmons,kmon,reid_kk,co2ij
    laispada = get_indata(fplai,laiallna,coreset,dalen,dalab,kmons,kmon,reid_kk); gc() 
    tasspada = get_indata(fptas,tasallna,coreset,dalen,dalab,kmons,kmon,reid_kk); gc() 
    prespada = get_indata(fppre,prallna ,coreset,dalen,dalab,kmons,kmon,reid_kk); gc() 
    
    redaout = calre_laitas(laispada, tasspada, prespada, cobsym_eo,reid_kk)
    
    setwd('/gpfs/')
    saveRDS(redaout, file = paste(outna, '_', kmon, sep=''))
    
    t2 = proc.time()
    print(paste(round((t2 - t1)[3]/60, 2), ' min'))
  }
}

setwd('/gpfs/')
cobsym = readRDS('gloe_c1')
pref(inna='gloe_c1', cobsym)
