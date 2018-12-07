# Unlike traditional test_train_split using random initialization, 
#here I am splitting the dataset into train and validation datasets in order. 
#Every sentence is broken down into multiple phrases, 
#and so a random split would ensure that starkly similar phrases 
#from the training set would land in the validation set, 
#thereby the validation set performance misleading us into believing that 
#the model generalized well, while all it did was encounter a validation dataset 
#that was mostly a subset of the training dataset.

def train_test_split(train_df_flags,df_train):
    X_train_tf = train_df_flags[0:125000]
    X_valid_tf = train_df_flags[125000:]
    y_train_tf = (df_train["Sentiment"])[0:125000]
    y_valid_tf = (df_train["Sentiment"])[125000:]
    print("X_train shape: ", X_train_tf.shape)
    print("X_valid shape: ",X_valid_tf.shape)
    print("Y_train shape: ",len(y_train_tf))
    print("Y_valid shape: ",len(y_valid_tf))
    return  X_train_tf,X_valid_tf,y_train_tf,y_valid_tf