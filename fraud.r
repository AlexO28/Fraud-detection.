library(data.table)
glueByName <- function(tab) {
  tab <- as.data.table(as.data.frame(tab))
  tabOfFrauds <- tab[type == 'TRANSFER', list(nameOrig, nameDest)]
  tab <- tab[(type != 'TRANSFER'), ]
  setkey(tab, "nameOrig")
  print(nrow(tab))
  setkey(tabOfFrauds, "nameOrig")
  print(nrow(tabOfFrauds[!is.na(nameDest),]))
  return(tabOfFrauds[tab,])
  badList <- unique(tab[(isFraud==1) && (type != 'TRANSFER'), nameOrig])
  print(length(badList))
  for (i in 1:nrow(tab)) {
    if (i %% 1000 == 0) {
	  print(i)
    }
	name <- tab$nameOrig[i]
	if (name %in% badList) {
	  tab[nameOrig == tab$nameDest[i], nameOrig := name]
	}
  }
  tab
}
getStatByNames <- function(tab) {
  tab <- as.data.table(as.data.frame(tab))
  setkey(tab, "step")
  print(head(tab))
  print(tail(tab))
  names <- unique(tab[, nameOrig])
  print(length(names))
  print(nrow(tab))
  vec1 <- numeric(length(names))
  vec2 <- numeric(length(names))
  vec3 <- numeric(length(names))
  vec4 <- numeric(length(names))
  for (i in 1:length(names)) {
	print(i)
	temptab <- as.data.frame(tab[nameOrig == names[i], ])
	#print(temptab)
    vec2[i] <- temptab$amount[2] - temptab$amount[1]
	vec1[i] <- nrow(temptab)
	vec3[i] <- temptab$step[2] - temptab$step[1]
	vec4[i] <- sum(temptab$isFraud)
	#print(vec4[i])
  }
  data.frame(cbind(names, vec1, vec2, vec3, vec4))
}
makeQuantileIntensivityPlot <- function(tab, numOfQuants) {
  numOfRows <- c()
  numOfFrauds <- c()
  quants1 <- c()
  quants2 <- c()
  for (i in 1:numOfQuants) {
    print(i)
    quant1 <- as.numeric(tab[, quantile(Val, (i-1)/numOfQuants)])
    quant2 <- as.numeric(tab[, quantile(Val, i/numOfQuants)])
	numOfRows <- c(numOfRows, nrow(tab[(Val >= quant1) & (Val < quant2), ]))
	numOfFrauds <- c(numOfFrauds, nrow(tab[(Class == 1) & (Val >= quant1) & (Val < quant2), ]))
	quants1 <- c(quants1, quant1)
	quants2 <- c(quants2, quant2)
  }
data.table(numOfRows = numOfRows, numOfFrauds = numOfFrauds, quants1 = quants1, quants2 = quants2)
}
prepareData <- function(q, avstat) {
  q$quants1[1] <- -1000
  q$quants2[nrow(q)] <- 1000
  q[, x := (quants1 + quants2)/2]
  q[, Val := numOfFrauds/numOfRows]
  q[, Val := Val/avstat]
q
}
checkModelBS <- function(tab, quanttab, avstat) {
  s <- 0
  s2 <- 0
  s3 <- 0
  s4 <- 0
  for (i in 1:nrow(quanttab)) {
    #print('***')
	#print(c(quanttab$quants1[i], quanttab$quants2[i]))
    tabred <- tab[(Val >= quanttab$quants1[i]) & (Val < quanttab$quants2[i]), ]
	prob1 <- quanttab$Val[i]*avstat
	#print(head(tabred))
	prob2 <- nrow(tabred[Class==1, ])/(nrow(tabred)+0.01)
	#print(c(prob1, prob2))
	s <- s + (prob1-prob2)^2
	s2 <- s2 + nrow(tabred[Class==0, ])*(prob1)^2 + nrow(tabred[Class==1, ])*(1- prob1)^2
	prob3 <- ifelse(prob1 >= min(0.5, 5*avstat), 1, 0)
	s3 <- s3 + nrow(tabred[Class==0, ])*(prob3)^2 + nrow(tabred[Class==1, ])*(1- prob3)^2
	s4 <- s4 + nrow(tabred[Class==1, ])*(1- prob3)^2
  }
list(s/nrow(tab), s2/nrow(tab), 1-s3/nrow(tab), 1-s4/nrow(tab[Class==1, ]))
}
Main <- function(tab, numOfQuants) {
  print('started')
  train_ind <- sample(seq_len(nrow(tab)), size = nrow(tab)*0.75, replace = FALSE)
  tabtrain <- tab[train_ind, ]
  tabtest <- tab[-train_ind, ]
  quanttab <- makeQuantileIntensivityPlot(tabtrain, numOfQuants)
  print(summary(quanttab))
  avstat <- nrow(tabtrain[Class==1, ])/nrow(tabtrain)
  print(avstat)
  quanttab2 <- prepareData(copy(quanttab), avstat)
  print(summary(quanttab2))
checkModelBS(tabtest, quanttab2, avstat)
}
trainTestSplit <- function(tab) {
  train_ind <- sample(seq_len(nrow(tab)), size = nrow(tab)*0.75, replace = FALSE)
  tabtrain <- tab[train_ind, ]
  tabtest <- tab[-train_ind, ]
list(train=tabtrain, test=tabtest)
}
MainTotModel <- function(tab1, tab2, tab3, numOfQuants1, numOfQuants2, numOfQuants3) {
  print('started')
  dat1 <- trainTestSplit(tab1)
  dat2 <- trainTestSplit(tab2)
  dat3 <- trainTestSplit(tab3)
  tabtrain1 <- dat1$train
  tabtrain2 <- dat2$train
  tabtrain3 <- dat3$train
  tabtest1 <- dat1$test
  tabtest2 <- dat2$test
  tabtest3 <- dat3$test
  quanttab1 <- makeQuantileIntensivityPlot(tabtrain1, numOfQuants1)
  quanttab2 <- makeQuantileIntensivityPlot(tabtrain2, numOfQuants2)
  quanttab3 <- makeQuantileIntensivityPlot(tabtrain3, numOfQuants3)
  avstat1 <- nrow(tabtrain1[Class==1, ])/nrow(tabtrain1)
  avstat2 <- nrow(tabtrain2[Class==1, ])/nrow(tabtrain2)
  avstat3 <- nrow(tabtrain3[Class==1, ])/nrow(tabtrain3)
  quanttab1 <- prepareData(copy(quanttab1), avstat1)
  quanttab2 <- prepareData(copy(quanttab2), avstat2)
  quanttab3 <- prepareData(copy(quanttab3), avstat3)
  quanttab <- makeCommonIntensTab(quanttab1, quanttab2, quanttab3)
  print(summary(quanttab))
  print(c(avstat1, avstat2, avstat3))
list(check1 = checkModelBS(tabtest1, quanttab, avstat1),
     check2 = checkModelBS(tabtest2, quanttab, avstat2),
	 check3 = checkModelBS(tabtest3, quanttab, avstat3))  
}
getProbFromQuantTab <- function(q1, q2, quanttab) {
  quanttabred <- quanttab[(quants1<=q1) & (quants2>=q2), ]
  print(nrow(quanttabred))
quanttabred[, mean(Val)]
}
makeCommonIntensTab <- function(quanttab1, quanttab2, quanttab3) {
  print(summary(quanttab1))
  print(summary(quanttab2))
  print(summary(quanttab3))
  allpoints <- unique(c(quanttab1$quants1, quanttab2$quants1, quanttab3$quants1))
  print(summary(allpoints))
  allpoints <- sort(allpoints)
  quanttab <- data.frame(quants1 = allpoints, quants2 = c(allpoints[2:length(allpoints)], 1000), Val = NA)
  for (i in 1:nrow(quanttab)) {
    q1 <- quanttab$quants1[i]
	q2 <- quanttab$quants2[i]
	quanttab$Val[i] <- (getProbFromQuantTab(q1, q2, quanttab1) + getProbFromQuantTab(q1, q2, quanttab2) + getProbFromQuantTab(q1, q2, quanttab3))/3
  }
quanttab
}
findValByX <- function(q, x0) {
  q[x >= x0, min(Val)]
}
makeAverageLine <- function(q1, q2, q3) {
  q1 <- prepareData(q1)
  q2 <- prepareData(q2)
  q3 <- prepareData(q3)  
  xx <- unique(c(q1$x, q2$x, q3$x))
  xx <- sort(xx)
  yy <- c()
  for (x in xx) {
    skip1 <- 1
	skip2 <- 1
	skip3 <- 1
	val1 <- findValByX(q1, x)
	val2 <- findValByX(q2, x)
	val3 <- findValByX(q3, x)
    if (val1==Inf) {val1 <- 0; skip1 <- 0}
	if (val2==Inf) {val2 <-0; skip2 <- 0}
    if (val3==Inf) {val3 <- 0; skip3 <- 0}
    yy <- c(yy, (val1 + val2 + val3)/(skip1+skip2+skip3)) 
  }
data.frame(xx, yy)
}
LogLoss <- function(x, y) {
  z <- numeric(length(x))
  for (i in 1:length(z)) {
     z[i] <- x[i]*log(y[i]) + (1-x[i])*log(1-y[i])
  }
z
}
BrierScore <- function(x, y) {
  z <- numeric(length(x))
  for (i in 1:length(z)) {
     z[i] <- (x[i]-y[i])^2
  }
z
}
CheckTotAvModel <- function(totquant, tab, avstat) {
  xprev <- -100
  numstat <- numeric(length(totquant))
  for (i in 1:nrow(totquant)) {
    xcur <- totquant$xx[i]
	tabred <- tab[(Val > xprev) & (Val < xcur), ]
	numstat[i] <- (nrow(tabred[Class==1,])/(nrow(tabred)))
	xprev <- xcur
  }
data.frame(V1=BrierScore(numstat, pmin(totquant$yy*avstat,1))/avstat)
}
PrepTabForAvModel <- function(totquant, tab, avstat) {
  tab$wf <- NULL
  tab[, wf := -1]
  for (i in 1:nrow(totquant)) {
    print(i)
    prob <- min(1, totquant$yy[i]*avstat)
	tab[(wf<0) & (Val < totquant$xx[i]), wf := 0.+as.numeric(prob)]
  }
  tab[wf<0, wf := totquant$yy[nrow(totquant)]*avstat]
  print(summary(tab))
  tab[, wf2 := ifelse(wf>=min(5*avstat, 0.5), 1, 0)]
  print(summary(tab))
  vec <- as.data.frame(BrierScore(tab$wf2, tab$Class))
  print(summary(vec))
  return(c(sum(BrierScore(tab$wf, tab$Class))/nrow(tab), 1-sum(vec)/nrow(tab), 1-sum(vec[tab$Class==1, ])/nrow(tab[Class==1,])))
}
GetIntensForTab <- function(tab, numOfQuants) {
  train <- sample(1:nrow(tab), nrow(tab)*0.75)
  tabtrain <- tab[train, ]
  tabtest <- tab[-train, ]
  quanttab <- makeQuantileIntensivityPlot(tabtrain, numOfQuants)
list(quant=quanttab, test=tabtest)
}
EncodeData <- function(tab) {
  tab <- copy(tab)
  #encode credit history
  tab[CreditHistory == 'A34', CreditHistory := -2]
  tab[CreditHistory == 'A33', CreditHistory := -1]
  tab[CreditHistory == 'A30', CreditHistory := 0]
  tab[CreditHistory == 'A32', CreditHistory := 1]
  tab[CreditHistory == 'A31', CreditHistory := 2]
  tab[, CreditHistory := as.numeric(CreditHistory)]
  #encode employment
  tab[Employment == 'A71', Employment := 0]
  tab[Employment == 'A72', Employment := 1]
  tab[Employment == 'A73', Employment := 2]
  tab[Employment == 'A74', Employment := 3]
  tab[Employment == 'A75', Employment := 4]
  tab[, Employment := as.numeric(Employment)]
  #encode guarantors info
  tab[Guarantors == 'A101', Guarantors := 0]
  tab[Guarantors == 'A102', Guarantors := 1]
  tab[Guarantors == 'A103', Guarantors := 2]
  tab[, Guarantors := as.numeric(Guarantors)]
  #encode job info
  tab[JobInfo == 'A171', JobInfo := 0]
  tab[JobInfo == 'A172', JobInfo := 1]
  tab[JobInfo == 'A173', JobInfo := 2]
  tab[JobInfo == 'A174', JobInfo := 3]
  tab[, JobInfo := as.numeric(JobInfo)]
  #encode telephone
  tab[Telephone == 'A191', Telephone := 0]
  tab[Telephone == 'A192', Telephone := 1]
  tab[, Telephone := as.numeric(Telephone)]
  #encode foreign
  tab[Foreign == 'A202', Foreign := 0]
  tab[Foreign == 'A201', Foreign := 1]
  tab[, Foreign := as.numeric(Foreign)]  
tab
}