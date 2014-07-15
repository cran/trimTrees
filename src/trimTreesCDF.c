#include <Rmath.h>
#include <R.h>

void zeroIntCDF(int *x, int length) {
    memset(x, 0, length * sizeof(int));
}

void zeroDoubleCDF(double *x, int length) {
    memset(x, 0, length * sizeof(double));
}

void trimTreesCDF(
          /* random forest inputs. */
          int *inbagCountSorted,
          int *termNodetrainSorted,
          int *ntree,
          double *ytrainSorted,
          int *ntrain,
          double *forestSupport,
          int *nSupport,
          int *termNodeNewX,
          double *ytest,
          int *ntest,
 
          /* user inputs. */
          double *trim,
          Rboolean *trimIsExterior,
          double *uQuantiles,
          int *nQuantiles,
          
          /* tree outputs. */
          double *treeValues,
          double *treeCounts,
          double *treeCumCounts,
          double *treeCDFs,
          double *treePMFs,
          double *treeMeans,
          double *treeVars,
          double *treePITs,
          double *treeQuantiles,
          double *treeFirstPMFValues,
          
          /* ensembles outputs. */
          double *bracketingRate,
          double *bracketingRateAllPairs,
          
          double *rfClassEnsembleCDFs,
          double *rfClassEnsembleQuantiles,
          double *rfClassEnsembleComponentScores,
          double *rfClassEnsembleScores,
          
          double *trimmedEnsembleCDFs,
          double *trimmedEnsemblePMFs,
          double *trimmedEnsembleMeans,
          double *trimmedEnsembleVars,
          double *trimmedEnsemblePITs,
          double *trimmedEnsembleQuantiles,
          double *trimmedEnsembleComponentScores,
          double *trimmedEnsembleScores,
          
          double *untrimmedEnsembleCDFs,
          double *untrimmedEnsemblePMFs,
          double *untrimmedEnsembleMeans,
          double *untrimmedEnsembleVars,
          double *untrimmedEnsemblePITs,
          double *untrimmedEnsembleQuantiles,
          double *untrimmedEnsembleComponentScores,
          double *untrimmedEnsembleScores
          ) 
  
  {

  int t, i, j, k, lo, hi, nTrim, *index, indexPIT;
  double *cdfValuesToTrim, trimmedSum;
  
  index = (int *) R_alloc(*nQuantiles, sizeof(int));
  cdfValuesToTrim = (double *) R_alloc(*ntree, sizeof(double));

  /* Set the low and high indices for trimming. */
  if(*trimIsExterior) {
    lo = (int)((*ntree) * (*trim));
    hi = *ntree - lo;
    if(lo == hi) { /* lo == hi when exterior trimming level is 0.5 and ntree is even. 
        In this case, the trimmed ensemble is the median forecast. */
      lo -= 1; 
      hi += 1; 
    }
    nTrim = *ntree - 2 * lo;
  } else {
    lo = (int)((0.5 - (*trim)) * (*ntree));
    if(lo == 0) lo += 1;
    hi = (*ntree) - lo + 1;
    nTrim = 2 * lo;
  }  

  /* Start big loop over the rows in the test set. */
  for(t = 1; t <= *ntest; t++) {
      
    /* ------------
       TREE OUTPUTS
       ------------ */

    /* This loop finds each tree's y values (not necessarily unique) that are both inbag and in the new X's terminal node.
       (Note that the y values in a training set may not be unique.)  
       This loop also finds each tree's counts and cumulative counts of these y values and listed them by the unique y values. */
    zeroDoubleCDF(treeValues, (*ntrain) * (*ntree));
    zeroDoubleCDF(treeCounts, (*nSupport) * (*ntree));
    zeroDoubleCDF(treeCumCounts, (*nSupport + 1) * (*ntree));
    zeroDoubleCDF(treeCDFs, (*nSupport + 1) * (*ntree));
    zeroDoubleCDF(treePMFs, (*nSupport) * (*ntree));
    for(i = 1; i <= *ntree; i++) {
      k = 1;
      for(j = 1; j <= *nSupport; j++) {
        while(forestSupport[j - 1] == ytrainSorted[k - 1]) {
          if(termNodetrainSorted[(i - 1) * (*ntrain) + k - 1] == termNodeNewX[(i - 1) * (*ntest) + t - 1] && inbagCountSorted[(i - 1) * (*ntrain) + k - 1] != 0) {
            treeValues[(i - 1) * (*ntrain) + k - 1] =  ytrainSorted[k - 1];     
            treeCounts[(i - 1) * (*nSupport) + j - 1] += inbagCountSorted[(i - 1) * (*ntrain) + k - 1];
          } else treeValues[(i-1) * (*ntrain) + k - 1] =  NA_REAL;
          k++;  
        }
        treeCumCounts[(i - 1) * (*nSupport + 1) + j] = treeCumCounts[(i - 1) * (*nSupport + 1) + j - 1] + treeCounts[(i - 1) * (*nSupport) + j - 1];      
      }
    }
  
    /* This loop finds the trees' cdfs. */
    for(j = 1; j <= *nSupport; j++) {
      for(i = 1; i <= *ntree; i++) {
        treeCDFs[(i - 1) * (*nSupport + 1) + j] =  treeCumCounts[(i - 1) * (*nSupport + 1) + j] / treeCumCounts[(i - 1) * (*nSupport  + 1) + *nSupport];
        treePMFs[(i - 1) * (*nSupport) + j - 1] =  treeCounts[(i - 1) * (*nSupport) + j - 1] / treeCumCounts[(i - 1) * (*nSupport  + 1) + *nSupport];
        if(j == 1) treeFirstPMFValues[(i - 1) * (*ntest) + t - 1] = treePMFs[(i - 1) * (*nSupport)];
      }
    }
            
    /*This loop calculates the trees' means and variances. */
    for(i = 1; i <= *ntree; i++) {
      for(j = 1; j <= *nSupport; j++) {
        treeMeans[(i - 1) * (*ntest) + t - 1] +=  treePMFs[(i - 1) * (*nSupport) + j - 1] * forestSupport[j - 1]; 
        treeVars[(i - 1) * (*ntest) + t - 1] +=  treePMFs[(i - 1) * (*nSupport) + j - 1] * forestSupport[j - 1] * forestSupport[j - 1];
      }
       treeVars[(i - 1) * (*ntest) + t - 1] -= (treeMeans[(i - 1) * (*ntest) + t - 1] * treeMeans[(i - 1) * (*ntest) + t - 1]);
    }
    
    /* This loop finds each tree's PIT.  */
    for(i = 1; i <= *ntree; i++) {
        if(ytest[t - 1] < forestSupport[0]) {
          treePITs[(i - 1) * (*ntest) + t - 1] = 0;
        } else {
        indexPIT = *nSupport;
        while(ytest[t - 1] < forestSupport[indexPIT - 1])  indexPIT -= 1;
        treePITs[(i - 1) * (*ntest) + t - 1] = treeCDFs[(i - 1) * (*nSupport + 1) + indexPIT];
        }
    }
    
    /* This loop finds the quantiles of each tree's cdf.  */  
    zeroDoubleCDF(treeQuantiles, (*ntree) * (*nQuantiles));
    for(i = 1; i <= *ntree; i++) {
      zeroIntCDF(index, *nQuantiles);
      for(k = *nQuantiles; k >= 1; k--) {
        if(k == *nQuantiles) index[k - 1] = *nSupport;
        else index[k - 1] = index[k];
        while(uQuantiles[k - 1] <= treeCDFs[(i - 1) * (*nSupport + 1) + index[k - 1]]) index[k - 1] -= 1;
        treeQuantiles[(k - 1) * (*ntree) + i - 1] = forestSupport[index[k - 1]];
        index[k - 1]++;
      }
    }
    
    /* This loop calculates the bracketing rate among the trees, for each test value.  */  
    double countAbove;
    countAbove = 0;
    for(i = 1; i <= *ntree; i++) {
      if(treeMeans[(i - 1) * (*ntest) + t - 1] > ytest[t - 1])
         countAbove ++;
      }  
    bracketingRate[t - 1] = 2* countAbove/ (double)(*ntree) *(1 - countAbove/ (double)(*ntree));
   
    /* This loop calculates the bracketing rate among each pair of trees, for each test value.  */
     for(i = 1; i <= (*ntree-1); i++){
       for(j = i + 1; j <= *ntree; j++){
        if(treeMeans[(i - 1) * (*ntest) + t - 1] <= ytest[t - 1] && ytest[t - 1] <= treeMeans[(j - 1) * (*ntest) + t - 1])
            bracketingRateAllPairs [(j - 1) * (*ntree) + i - 1] ++;
        if(treeMeans[(i - 1) * (*ntest) + t - 1] >= ytest[t - 1] && ytest[t - 1] >= treeMeans[(j - 1) * (*ntest) + t - 1])
            bracketingRateAllPairs [(j - 1) * (*ntree) + i - 1] ++;
         }
      } 

    /* --------------------------
       UNTRIMMED ENSEMBLE OUTPUTS
       -------------------------- */

    /* This loop finds the untrimmed ensemble's cdf. */
    for(j = 1; j <= *nSupport; j++) {
      for(i = 1; i <= *ntree; i++) {
      untrimmedEnsembleCDFs[j * (*ntest) +  t - 1] += treeCDFs[(i - 1) * (*nSupport + 1) + j] / (double)(*ntree);
      }
      untrimmedEnsemblePMFs[(j - 1) * (*ntest) +  t - 1] = untrimmedEnsembleCDFs[j * (*ntest) +  t - 1] - untrimmedEnsembleCDFs[(j - 1) * (*ntest) +  t - 1];
    }  
    
    /* This loop finds the untrimmed ensemble's mean and variance. */
    for(i = 1; i <= *ntree; i++) {
      untrimmedEnsembleMeans[t - 1] += treeMeans[(i - 1) * (*ntest) + t - 1] / (double)(*ntree);
    }
    
    
    for(i = 1; i <= *ntree; i++) {
      untrimmedEnsembleVars[t - 1] += (treeMeans[(i - 1) * (*ntest) + t - 1] - untrimmedEnsembleMeans[t - 1]) 
                                      * (treeMeans[(i - 1) * (*ntest) + t - 1] - untrimmedEnsembleMeans[t - 1]) / (double)(*ntree)
                                      + treeVars[(i - 1) * (*ntest) + t - 1] / (double)(*ntree);
    }
       
    /* This statement finds the untrimmed ensemble's PIT. */
   if(ytest[t - 1] < forestSupport[0]) {
      untrimmedEnsemblePITs[t - 1] = 0;
    } else {
      indexPIT = *nSupport;
      while(ytest[t - 1] < forestSupport[indexPIT - 1])  indexPIT -= 1;
      untrimmedEnsemblePITs[t - 1] = untrimmedEnsembleCDFs[indexPIT * (*ntest) +  t - 1];
    }    
    
    
    /* This loop finds the untrimmed ensemble's quantiles.  */
    zeroIntCDF(index, *nQuantiles);
    zeroDoubleCDF(untrimmedEnsembleQuantiles, *nQuantiles);
    for(k = *nQuantiles; k >= 1; k--) {
      if(k == *nQuantiles) index[k - 1] = *nSupport;
      else index[k - 1] = index[k];
      while(uQuantiles[k - 1] <= untrimmedEnsembleCDFs[index[k - 1] * (*ntest) + t - 1]) index[k - 1] -= 1;
      untrimmedEnsembleQuantiles[k - 1] = forestSupport[index[k - 1]];
      index[k - 1]++;
    }  

    /* These loops find the untrimmed ensemble's scores: LinQuanS, LogQuanS, RPS and TMS. */
    zeroDoubleCDF(untrimmedEnsembleComponentScores, *nQuantiles*2);
    for(k = 1; k <= *nQuantiles; k++) {
      if(untrimmedEnsembleQuantiles[k - 1] <= ytest[t - 1]) 
        untrimmedEnsembleComponentScores[k - 1] = - uQuantiles[k - 1] * (ytest[t - 1] - untrimmedEnsembleQuantiles[k - 1]);
      else 
        untrimmedEnsembleComponentScores[k - 1] = - (1 - uQuantiles[k - 1]) * (untrimmedEnsembleQuantiles[k - 1] - ytest[t - 1]);
      untrimmedEnsembleScores[t - 1] += untrimmedEnsembleComponentScores[k - 1];   
      }
    
    for(k = 1; k <= *nQuantiles; k++) {
      if(untrimmedEnsembleQuantiles[k - 1] <= ytest[t - 1]) 
        untrimmedEnsembleComponentScores[*nQuantiles + k - 1] = - uQuantiles[k - 1] * (log(ytest[t - 1]) - log(untrimmedEnsembleQuantiles[k - 1]));
      else 
        untrimmedEnsembleComponentScores[*nQuantiles + k - 1] = - (1 - uQuantiles[k - 1]) * (log(untrimmedEnsembleQuantiles[k - 1]) - log(ytest[t - 1]));
      untrimmedEnsembleScores[(*ntest) + t - 1] += untrimmedEnsembleComponentScores[*nQuantiles + k - 1];   
      }
    
    for(j = 1; j <= *nSupport - 1; j++) {
      if(forestSupport[j - 1] < ytest[t - 1])
        untrimmedEnsembleScores[2 * (*ntest) + t - 1] -= untrimmedEnsembleCDFs[j * (*ntest) +  t - 1] * untrimmedEnsembleCDFs[j * (*ntest) +  t - 1];
      else 
        untrimmedEnsembleScores[2 * (*ntest) + t - 1] -= (1 - untrimmedEnsembleCDFs[j * (*ntest) +  t - 1]) * (1 - untrimmedEnsembleCDFs[j * (*ntest) +  t - 1]);
      }    
    
    untrimmedEnsembleScores[3 * (*ntest) + t - 1] = log(1/untrimmedEnsembleVars[t - 1]) - (1/untrimmedEnsembleVars[t - 1]) * (ytest[t - 1]-untrimmedEnsembleMeans[t - 1]) * (ytest[t - 1]-untrimmedEnsembleMeans[t - 1]);    
   

    /* ------------------------
       TRIMMED ENSEMBLE OUTPUTS
       ------------------------ */  
    /* This loop finds the trimmed ensemble's cdf. */
    zeroDoubleCDF(cdfValuesToTrim, *ntree);
    if((*trim < 0.000001) && (*trim > -0.000001)){
      for(j = 1; j <= *nSupport; j++) {
        trimmedEnsembleCDFs[j * (*ntest) +  t - 1] = untrimmedEnsembleCDFs[j * (*ntest) +  t - 1];
        trimmedEnsemblePMFs[(j - 1) * (*ntest) +  t - 1] = trimmedEnsembleCDFs[j * (*ntest) +  t - 1] - trimmedEnsembleCDFs[(j - 1) * (*ntest) +  t - 1];
      }
    }
    else {
      for(j = 1; j <= *nSupport; j++) {
        for(i = 1; i <= *ntree; i++) {
          cdfValuesToTrim[i - 1] = treeCDFs[(i - 1) * (*nSupport + 1) + j];
        }
        if(*trimIsExterior) {
          R_rsort(cdfValuesToTrim, *ntree);
          trimmedSum = 0; 
          for(k = lo; k < hi; k++) trimmedSum += cdfValuesToTrim[k];
          trimmedEnsembleCDFs[j * (*ntest) +  t - 1] = trimmedSum / (double)(nTrim);
        } else {
            R_rsort(cdfValuesToTrim, *ntree);
            trimmedSum = 0;
            for(k = 0; k < lo; k++) trimmedSum += cdfValuesToTrim[k];
            for(k = hi - 1; k < *ntree; k++) trimmedSum += cdfValuesToTrim[k];
            trimmedEnsembleCDFs[j * (*ntest) +  t - 1] = trimmedSum / (double)(nTrim);
        }
        trimmedEnsemblePMFs[(j - 1) * (*ntest) +  t - 1] = trimmedEnsembleCDFs[j * (*ntest) +  t - 1] - trimmedEnsembleCDFs[(j - 1) * (*ntest) +  t - 1];
      }  
    }
    
    /* This loop finds the trimmed ensemble's mean and variance.  */
     for(j = 1; j <= *nSupport; j++) {
      trimmedEnsembleMeans[t - 1] += trimmedEnsemblePMFs[(j - 1) * (*ntest) +  t - 1] * forestSupport[j - 1];
      trimmedEnsembleVars[t - 1] += trimmedEnsemblePMFs[(j - 1) * (*ntest) +  t - 1] * forestSupport[j - 1] * forestSupport[j - 1];
      }
      trimmedEnsembleVars[t - 1] -= trimmedEnsembleMeans[t - 1] * trimmedEnsembleMeans[t - 1];
    
        
    /* This statement finds the trimmed ensemble's PIT. */
    if(ytest[t - 1] < forestSupport[0]) {
      trimmedEnsemblePITs[t - 1] = 0;
    } else {
      indexPIT = *nSupport;
      while(ytest[t - 1] < forestSupport[indexPIT - 1])  indexPIT -= 1;
      trimmedEnsemblePITs[t - 1] = trimmedEnsembleCDFs[indexPIT * (*ntest) +  t - 1];
    }
      
    /* This loop finds the trimmed ensemble's quantiles. */ 
    zeroIntCDF(index, *nQuantiles);
    zeroDoubleCDF(trimmedEnsembleQuantiles, *nQuantiles);
    for(k = *nQuantiles; k >= 1; k--) {
      if(k == *nQuantiles) index[k - 1] = *nSupport;
      else index[k - 1] = index[k];
      while(uQuantiles[k - 1] <= trimmedEnsembleCDFs[index[k - 1] * (*ntest) + t - 1]) index[k - 1] -= 1;
      trimmedEnsembleQuantiles[k - 1] = forestSupport[index[k - 1]];
      index[k - 1]++;
    } 
  
    /* These loops find the trimmed ensemble's scores: LinQuanS, LogQuanS, RPS and TMS. */
    zeroDoubleCDF(trimmedEnsembleComponentScores, *nQuantiles*2);
    for(k = 1; k <= *nQuantiles; k++) {
      if(trimmedEnsembleQuantiles[k - 1] <= ytest[t - 1]) 
        trimmedEnsembleComponentScores[k - 1] = - uQuantiles[k - 1] * (ytest[t - 1] - trimmedEnsembleQuantiles[k - 1]);
      else 
        trimmedEnsembleComponentScores[k - 1] = - (1 - uQuantiles[k - 1]) * (trimmedEnsembleQuantiles[k - 1] - ytest[t - 1]);
      trimmedEnsembleScores[t - 1] += trimmedEnsembleComponentScores[k - 1];   
      }
        
    for(k = 1; k <= *nQuantiles; k++) {
      if(trimmedEnsembleQuantiles[k - 1] <= ytest[t - 1]) 
        trimmedEnsembleComponentScores[*nQuantiles + k - 1] = - uQuantiles[k - 1] * (log(ytest[t - 1]) - log(trimmedEnsembleQuantiles[k - 1]));
      else 
        trimmedEnsembleComponentScores[*nQuantiles + k - 1] = - (1 - uQuantiles[k - 1]) * (log(trimmedEnsembleQuantiles[k - 1]) - log(ytest[t - 1]));
      trimmedEnsembleScores[(*ntest) + t - 1] += trimmedEnsembleComponentScores[*nQuantiles + k - 1];   
      }
    
    for(j = 1; j <= *nSupport - 1; j++) {
      if(forestSupport[j - 1] < ytest[t - 1])
        trimmedEnsembleScores[2 * (*ntest) + t - 1] -= trimmedEnsembleCDFs[j * (*ntest) +  t - 1] * trimmedEnsembleCDFs[j * (*ntest) +  t - 1];
      else 
        trimmedEnsembleScores[2 * (*ntest) + t - 1] -= (1 - trimmedEnsembleCDFs[j * (*ntest) +  t - 1]) * (1 - trimmedEnsembleCDFs[j * (*ntest) +  t - 1]);
      }  
      
      trimmedEnsembleScores[3 * (*ntest) + t - 1] = log(1/trimmedEnsembleVars[t - 1]) - (1/trimmedEnsembleVars[t - 1]) * (ytest[t - 1]-trimmedEnsembleMeans[t - 1]) * (ytest[t - 1]-trimmedEnsembleMeans[t - 1]);
    
   /* --------------------------------
       RANDOM FOREST'S ENSEMBLE OUTPUTS
       -------------------------------- */
    
    /* This loop finds the random forest's ensemble quantiles. */ 
    zeroIntCDF(index, *nQuantiles);
    zeroDoubleCDF(rfClassEnsembleQuantiles, *nQuantiles);
    for(k = *nQuantiles; k >= 1; k--) {
      if(k == *nQuantiles) index[k - 1] = *nSupport;
      else index[k - 1] = index[k];
      while(uQuantiles[k - 1] <= rfClassEnsembleCDFs[index[k - 1] * (*ntest) + t - 1]) index[k - 1] -= 1;
      rfClassEnsembleQuantiles[k - 1] = forestSupport[index[k - 1]];
      index[k - 1]++;
    } 
  
    /* This loop finds the rf's score. LinQuanS, LogQuanS, RPS, TMS. */
    zeroDoubleCDF(rfClassEnsembleComponentScores, *nQuantiles*2);
    for(k = 1; k <= *nQuantiles; k++) {
      if(rfClassEnsembleQuantiles[k - 1] <= ytest[t - 1]) 
         rfClassEnsembleComponentScores[k - 1] = - uQuantiles[k - 1] * (ytest[t - 1] - rfClassEnsembleQuantiles[k - 1]);
      else 
        rfClassEnsembleComponentScores[k - 1] = - (1 - uQuantiles[k - 1]) * (rfClassEnsembleQuantiles[k - 1] - ytest[t - 1]);
       rfClassEnsembleScores[t - 1] += rfClassEnsembleComponentScores[k - 1];   
      }
    
    for(k = 1; k <= *nQuantiles; k++) {
        if(rfClassEnsembleQuantiles[k - 1] <= ytest[t - 1]) 
          rfClassEnsembleComponentScores[*nQuantiles + k - 1] = - uQuantiles[k - 1] * (log(ytest[t - 1]) - log(rfClassEnsembleQuantiles[k - 1]));
        else 
          rfClassEnsembleComponentScores[*nQuantiles + k - 1] = - (1 - uQuantiles[k - 1]) * (log(rfClassEnsembleQuantiles[k - 1]) - log(ytest[t - 1]));
        rfClassEnsembleScores[(*ntest) + t - 1] += rfClassEnsembleComponentScores[*nQuantiles + k - 1];   
      }
    
    for(j = 1; j <= *nSupport - 1; j++) {
        if(forestSupport[j - 1] < ytest[t - 1])
          rfClassEnsembleScores[2 * (*ntest) + t - 1] -= rfClassEnsembleCDFs[j * (*ntest) +  t - 1] * rfClassEnsembleCDFs[j * (*ntest) +  t - 1];
        else 
          rfClassEnsembleScores[2 * (*ntest) + t - 1] -= (1 - rfClassEnsembleCDFs[j * (*ntest) +  t - 1]) * (1 - rfClassEnsembleCDFs[j * (*ntest) +  t - 1]);
      }      

  }  /* End of loop over test rows. */


} /* End of function. */




  
