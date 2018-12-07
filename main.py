import pre_processing
import train_test_split as tts
import models
import generate_csv

def main():

    #exploratory Data Analysis can be found in eda.py.
    
    #Data Preprocessing
    prp = pre_processing.Pre_Processing() 
    df_train1,df_test = prp.read_file() #read files
    df_train = prp.drop_unecessary_columns(df_train1) #drop unnecessary columns
    df_train['Phrase'] = [prp.cleaning(df_train,s) for s in df_train['Phrase']] #clean it of not needed chracters
    sentences2 = prp.stem_lemmatize(df_train,df_test) #stem and lemmatize 
    
    train_df_flags,test_df_flags = prp.tfidf(df_train,df_test,sentences2) #tfidf

    X_train_tf,X_valid_tf,y_train_tf,y_valid_tf = tts.train_test_split(train_df_flags,df_train) #train_test_split

    m = models.Model() #create a model object

    #logistic regression
    clf = m.logistic_regression(X_train_tf,X_valid_tf,y_train_tf,y_valid_tf)
    generate_csv.generate(df_test,test_df_flags,clf,'lrx')
    
    #svc
    sv = m.svc_(X_train_tf,X_valid_tf,y_train_tf,y_valid_tf)
    generate_csv.generate(df_test,test_df_flags,sv,'sv')
    
    #random_forest_classifier
    rf = m.random_forest(X_train_tf,X_valid_tf,y_train_tf,y_valid_tf)
    generate_csv.generate(df_test,test_df_flags,clf,'rfrst')
    
    #ada_boost
    ab = m.ada_boost(X_train_tf,X_valid_tf,y_train_tf,y_valid_tf)
    generate_csv.generate(df_test,test_df_flags,clf,'abst')
    
if __name__ == '__main__':
    main()