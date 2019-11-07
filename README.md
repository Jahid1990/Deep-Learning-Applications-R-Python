# Telecom Products-Recommendation-Engine-Using-Deep-Learning in R

#Packages Required:
1. Tensorflow # Backend of Keras
2. Keras # Deep learning framework
3. Car # For recode purpose
4. RJDBC # Connect R with database. This can be avoided by loading data manually from other source rather than database.

# Data Loading
# Database Approach
drv <- JDBC("oracle.jdbc.OracleDriver",classPath="D:/Jahid/ojdbc6.jar", " ")

con <- dbConnect(drv, "jdbc:oracle:thin:@//edw-scan:1521/database name","your data base schema name","your own credentials")

data<-dbGetQuery(con, "select * FROM TMP_KPI") #TMP_PACK_REC_FINAL

# Manual Approach
data<-read.csv("filepath/data.csv",sep=",")

data[is.na(data)]<-0

# Train & Test Dataset Preparation
indices<-sample(1:nrow(data),nrow(data)*.8)
train<-data[indices,]
test<-data[-indices,]

x_train<-as.matrix(train[,-c(1,15,16)])
x_train<-scale(x_train)
y_train<-train[,c(15,16)]

#24,44,53,54,89,99,108,117,148,199,239,288,289

r<-recode(y_train[,1], "0=0; 24=1;44=2;53=3;54=4;89=5;99=6;108=7;117=8;148=9;199=10;239=11;288=12;289=13")

y_train1<-to_categorical(r,num_classes =NULL)
y_train2<-to_categorical(y_train[,2],num_classes =NULL)
y_train<-as.matrix(data.frame(data.frame(y_train1),y_train2))
column<-c(paste("pack",c(0,24,44,53,54,89,99,108,117,148,199,239,288,289),sep="_"),paste("day",c(0:31),sep="_"))
colnames(y_train)<-column
head(y_train)

x_test<-as.matrix(test[,-c(1,15,16)])
x_test<-scale(x_test)
y_test<-test[,c(15,16)]

#24,44,53,54,89,99,108,117,148,199,239,288,289

r<-recode(y_test[,1], "0=0; 24=1;44=2;53=3;54=4;89=5;99=6;108=7;117=8;148=9;199=10;239=11;288=12;289=13")

y_test1<-to_categorical(r,num_classes =NULL)
y_test2<-to_categorical(y_test[,2],num_classes =NULL)
y_test<-as.matrix(data.frame(data.frame(y_test1),y_test2))
column<-c(paste("pack",c(0,24,44,53,54,89,99,108,117,148,199,239,288,289),sep="_"),paste("day",c(0:31),sep="_"))
colnames(y_test)<-column


inputs <- layer_input(shape = c(42))
m<-layer_dense(units = 512, activation = 'relu') (inputs)
m<-layer_dense(units = 512, activation = 'relu') (m)
m<-layer_dense(units = 512, activation = 'relu') (m)
m<-layer_dense(units = 512, activation = 'relu') (m)
m<-layer_dense(units = 512, activation = 'relu') (m)
m<-layer_dense(units = 512, activation = 'relu') (m)
m<-layer_dense(units = 512, activation = 'relu') (m)
m<-layer_dense(units = 512, activation = 'relu') (m)
m<-layer_dense(units = 512, activation = 'relu') (m)

pack<-layer_dense(units =512,activation ='relu')(m)
pack<-layer_dense(units =512,activation ='relu')(m)
pack<-layer_dense(units =512,activation ='relu')(m)
pack<-layer_dense(units =512,activation ='relu')(m)
pack<-layer_dense(units =256,activation ='relu')(pack)
pack<-layer_dense(units =256,activation ='relu')(pack)
pack<-layer_dense(units =128,activation ='relu')(pack)
pack<-layer_dense(units =512,activation ='relu')(pack)
pack<-layer_dense(units =256,activation ='relu')(pack)
pack<-layer_dense(units =256,activation ='relu')(pack)
pack<-layer_dense(units =64,activation ='relu')(pack)
pack<-layer_dense(units =14,activation ='softmax',name="packs")(pack)

day<-layer_dense(units =512,activation ='relu')(m)
day<-layer_dense(units =512,activation ='relu')(m)
day<-layer_dense(units =512,activation ='relu')(m)
day<-layer_dense(units =512,activation ='relu')(m)
day<-layer_dense(units =256,activation ='relu')(day)
day<-layer_dense(units =128,activation ='relu')(day)
day<-layer_dense(units =128,activation ='relu')(day)
day<-layer_dense(units =128,activation ='relu')(day)
day<-layer_dense(units =128,activation ='relu')(day)
day<-layer_dense(units =64,activation ='relu')(day)
day<-layer_dense(units =32,activation ='softmax',name = "days")(day)



conc<-layer_concatenate(c(pack,day))
model <- keras_model(inputs = inputs, outputs =list(pack,day))

summary(model)

model %>% compile(
  loss=list("categorical_crossentropy","categorical_crossentropy"),
  optimizer = optimizer_adam(lr =10^-4, beta_1 = 0.9, beta_2 = 0.999,
                             epsilon =10^-9, decay = 0, amsgrad = FALSE, clipnorm = NULL,
                             clipvalue = NULL),  metrics = c("categorical_accuracy")
)




history <-model %>% fit(
  x_train, list(y_train[,1:14],y_train[,15:46]),
  batch_size =30000,
  epochs =20,
  verbose = 1,
  validation_split = 0,
  validation_data = list(x_test,list(y_test[,1:14],y_test[,15:46]))
)




