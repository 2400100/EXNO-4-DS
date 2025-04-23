# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

      import pandas as pd
      import numpy as np
      import seaborn as sns
      
      from sklearn.model_selection import train_test_split
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.metrics import accuracy_score, confusion_matrix
      
      data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
      data
![322338017-b544c435-1cc1-4bc6-83c9-de2945348808](https://github.com/user-attachments/assets/a802c77d-c73a-4b9b-8e28-253e8f4ca294)


          data.isnull().sum()

![322338037-40b1ab98-5a1a-41a1-b943-102b7c4cabed](https://github.com/user-attachments/assets/35e62f57-6187-446f-a284-8380f3f15618)


      
         missing=data[data.isnull().any(axis=1)]
         missing  
![322338066-a5fe88ab-c993-4c97-b249-cffea5a21a54](https://github.com/user-attachments/assets/799cdcb2-fbb2-42bb-b6dc-36fac544a521)
 
     data2=data.dropna(axis=0)
     data2
![322338086-40a10680-63a6-4f18-87ae-517ceda76ca9](https://github.com/user-attachments/assets/1671e365-f9c0-4d8e-8029-d46d156d0bc4)

     sal=data["SalStat"]
     
     data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
     print(data2['SalStat'])
![322338114-e59ce957-1bdc-4455-97a5-15d66108b864](https://github.com/user-attachments/assets/e311cd17-2464-4f45-bc07-f3ca6221285a)

     sal2=data2['SalStat']
     
     dfs=pd.concat([sal,sal2],axis=1)
     dfs
![322338170-f8435063-835b-4eba-af2e-c46c67ea55e9](https://github.com/user-attachments/assets/357e152e-abb3-4c9b-80c6-9fc52320ee1c)



     data2
![322338187-c034e83a-8e21-400e-bc40-103e3da86d0e](https://github.com/user-attachments/assets/af6f9952-9d64-4107-ad10-3ec8f0d12b2c)

      new_data=pd.get_dummies(data2, drop_first=True)
      new_data
   
![322338213-f21819e3-a5bd-47e6-b1b7-9bc08b64bed9](https://github.com/user-attachments/assets/12062cbc-a44f-4871-a182-99296996be66)

          
         columns_list=list(new_data.columns)
         print(columns_list)
![322338249-8af6f5ce-4d99-4ed6-9371-730aeaa5a56b](https://github.com/user-attachments/assets/7c851cea-cc06-4b42-b3c8-9bb0daf498c8)
         

         features=list(set(columns_list)-set(['SalStat']))
         print(features)

![322338261-5f31a677-7d30-417a-8044-d5db741cafbf](https://github.com/user-attachments/assets/75ad76c9-bb60-42ac-b860-1e9229f45b19)

          y=new_data['SalStat'].values
          print(y)
![322338286-f4c779af-4c87-449e-9daa-be5d8d275212](https://github.com/user-attachments/assets/2be76bf4-6901-4199-8ac6-7bf86745fd12)

           
         x=new_data[features].values
         print(x)

![322338321-4154db03-4c87-4b98-a13b-964f19bee9b0](https://github.com/user-attachments/assets/33e94c97-f316-43ad-9fd2-584ef404609f)


        
         train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
         
         KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
         
         KNN_classifier.fit(train_x,train_y)
         
![322338343-e5e02520-eb39-436c-ac2e-e43048c1d672](https://github.com/user-attachments/assets/4ba847de-5afd-4efb-b018-e68b9e8ebae6)

         
        prediction=KNN_classifier.predict(test_x)
        
        confusionMatrix=confusion_matrix(test_y, prediction)
        print(confusionMatrix)

![322338371-a6eedfe3-aedd-4500-958f-6faafd54f464](https://github.com/user-attachments/assets/c70a617e-6a17-4c5c-8903-7b5cac33db18)


         
     accuracy_score=accuracy_score(test_y,prediction)
     print(accuracy_score)

![322338387-0e56ff41-2f35-4d01-b479-53547391567b](https://github.com/user-attachments/assets/6af53890-94a4-4d35-936b-60d73378c326)

          
     print("Misclassified Samples : %d" % (test_y !=prediction).sum())
![322338405-4af5ed3f-362a-40c6-a438-c89f31584e51](https://github.com/user-attachments/assets/87877b1b-855f-48c7-b924-caa6b38db1e5)


          
      data.shape
![322338420-1986f990-26e6-4b42-acfc-b2a6e52f8042](https://github.com/user-attachments/assets/391c6c0e-e2cd-4cad-acdf-18e2bfe0eccc)


      import pandas as pd
      import numpy as np
      from scipy.stats import chi2_contingency
      
      import seaborn as sns
      tips=sns.load_dataset('tips')
      tips.head()
      

![322338479-6d6f7ff2-b1da-4568-9cd1-cb6fa9553cd6](https://github.com/user-attachments/assets/b87a3eb4-d1fb-4029-9602-851c1cd43bbc)


     tips.time.unique()
![322338497-f77bc757-8a31-4a5d-be15-5a447e6549c6](https://github.com/user-attachments/assets/e15fe089-897a-4f5e-8247-b6017a7ea836)
       
     contingency_table=pd.crosstab(tips['sex'],tips['time'])
     print(contingency_table)

     
![322338518-06365e9f-f51b-4cf6-ab04-8a136726a025](https://github.com/user-attachments/assets/de001738-2e6f-4446-8a64-c8f0c860bdbe)

        
       chi2,p,_,_=chi2_contingency(contingency_table)
       print(f"Chi-Square Statistics: {chi2}")
       print(f"P-Value: {p}")

![322338543-6adc4da7-421c-458f-9ec6-f6158aa6f731](https://github.com/user-attachments/assets/4f0faf20-e79e-4fe6-b3b1-7e2ffdbdce09)

# RESULT:
       
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
