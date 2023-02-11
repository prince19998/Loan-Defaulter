#!/usr/bin/env python
# coding: utf-8
Capstone project

Prince Gupta
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns



import warnings
warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[2]:


df=pd.read_csv('Loan_Defaulter.csv')

pd.set_option('display.max_columns',None)

df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.describe().T


# In[8]:


df.isnull().sum()


# In[9]:


df.isna().apply(pd.value_counts).T


# In[10]:


df.isna().any()


# #  start ML

# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


le=LabelEncoder()


# In[13]:


df['Client_Education']=le.fit_transform(df['Client_Education'])
df['Accompany_Client']=le.fit_transform(df['Accompany_Client'])
df['Client_Income_Type']=le.fit_transform(df['Client_Income_Type'])
df['Client_Marital_Status']=le.fit_transform(df['Client_Marital_Status'])
df['Client_Gender']=le.fit_transform(df['Client_Gender'])
df['Loan_Contract_Type']=le.fit_transform(df['Loan_Contract_Type']) 
df['Client_Housing_Type']=le.fit_transform(df['Client_Housing_Type'])
df['Client_Occupation']=le.fit_transform(df['Client_Occupation'])
df['Client_Permanent_Match_Tag']=le.fit_transform(df['Client_Permanent_Match_Tag'])
df['Client_Contact_Work_Tag']=le.fit_transform(df['Client_Contact_Work_Tag'])
df['Type_Organization']=le.fit_transform(df['Type_Organization'])




df.head()


# In[ ]:





# In[14]:


df.head()


# In[15]:


a = df.isnull().sum()
perc = (a/(len(df)))*100
perc = pd.DataFrame(perc,columns=['Age of missing data'])
perc


# In[16]:


df['Client_Income'] = df['Client_Income'].replace('$',np.nan).astype('float')


# In[17]:


df['Bike_Owned']=df['Bike_Owned'].astype(float)


# In[18]:


df['Active_Loan'] = df['Active_Loan'].astype(float)


# In[19]:


df['House_Own'] = df['House_Own'].astype(float)


# In[20]:


df['Child_Count'] = df['Child_Count'].astype(float)


# In[21]:


df['Credit_Amount'] = df['Credit_Amount'].replace('$',np.nan).astype('float')


# In[22]:


df['Loan_Annuity'] = df['Loan_Annuity'].replace(['$','#VALUE!'],np.nan).astype('float')


# In[23]:


df['Accompany_Client'] = df['Accompany_Client'].replace(['Alone','Relative'
                                                         ,'Others','Kids','Partner','##','Group'],np.nan).astype('float')


# In[24]:


df['Client_Income_Type'] = df['Client_Income_Type'].replace(['Commercial','Service'
                                                             ,'Retired','Govt Job','Student'
                                                             ,'Unemployed','Maternity leave'
                                                             ,'Businessman'],np.nan).astype(float)


# In[25]:


df['Client_Education'] = df['Client_Education'].replace(['Secondary','Graduation'
                                                         ,'Graduation dropout','Junior secondary'
                                                         ,'Post Grad'],np.nan).astype(float)


# In[26]:


df['Client_Marital_Status'] = df['Client_Marital_Status'].replace(['M','W','S','D'],np.nan).astype('float')


# In[27]:


df['Client_Gender'] = df['Client_Gender'].replace(['Male','Female','XNA'],np.nan).astype(float)


# In[28]:


df['Loan_Contract_Type'] = df['Loan_Contract_Type'].replace(['CL','RL'],np.nan).astype(float)


# In[29]:


df['Client_Housing_Type'] = df['Client_Housing_Type'].replace(['Home','Family','Office'
                                                               ,'Municipal','Rental','Shared'],np.nan).astype(float)


# In[30]:


df['Population_Region_Relative'] = df['Population_Region_Relative'].replace(['@','#'],np.nan).astype(float)


# In[31]:


df['Age_Days'] = df['Age_Days'].replace(['X','x'],np.nan).astype(float)


# In[32]:


df['Employed_Days'] = df['Employed_Days'].replace(['x'],np.nan).astype(float)


# In[33]:


df['Registration_Days'] = df['Registration_Days'].replace(['x'],np.nan).astype(float)


# In[34]:


df['ID_Days'] = df['ID_Days'].replace(['x'],np.nan).astype(float)


# In[35]:


df['Client_Occupation'] = df['Client_Occupation'].replace(['Sales','Realty agents'
                                                           ,'Laborers','Core','Drivers','Managers','Accountants'
                                                           ,'High skill tech','Cleaning','HR','Waiters/barmen'
                                                           ,'Low-skill Laborers','Medicine','Cooking','Private service'
                                                           ,'Security','IT','Secretaries'],np.nan).astype(float)


# In[36]:


df['Client_Permanent_Match_Tag'] = df['Client_Permanent_Match_Tag'].replace(['Yes','No'],np.nan).astype(float)


# In[37]:


df['Client_Contact_Work_Tag'] = df['Client_Contact_Work_Tag'].replace(['Yes','No'],np.nan).astype(float)


# In[38]:


df['Type_Organization'] = df['Type_Organization'].replace(['Self-employed','Government'
                                                           ,'XNA','Business Entity Type 3','Other'
                                                           ,'Industry: type 3','Business Entity Type 2'
                                                           ,'Business Entity Type 1','Transport: type 4'
                                                           ,'Construction','Kindergarten','Trade: type 3'
                                                           ,'Industry: type 2','Trade: type 7','Trade: type 2'
                                                           ,'Agriculture','Military','Medicine','Housing'
                                                           ,'Industry: type 1','Industry: type 11'
                                                           ,'Bank','School','Industry: type 9'
                                                           ,'Postal','University','Transport: type 2'
                                                           ,'Restaurant','Electricity','Police'
                                                           ,'Industry: type 4','Security Ministries'
                                                           ,'Services','Transport: type 3','Mobile'
                                                           ,'Hotel','Security','Industry: type 7'
                                                           ,'Advertising','Cleaning','Realtor'
                                                           ,'Trade: type 6','Culture','Industry: type 5'
                                                           ,'Telecom','Trade: type 1','Industry: type 12'
                                                           ,'Industry: type 8','Insurance','Emergency'
                                                           ,'Legal Services','Industry: type 10'
                                                           ,'Trade: type 4','Industry: type 6'
                                                           ,'Transport: type 1','Industry: type 13'
                                                           ,'Religion','Trade: type 5'],np.nan).astype(float)


# In[39]:


df['Score_Source_3'] = df['Score_Source_3'].replace(['&'],np.nan).astype(float)


# In[40]:


df['Client_Income'].fillna(df['Client_Income'].median(),inplace=True)


# In[41]:


df['Bike_Owned'].fillna(df['Bike_Owned'].median(),inplace=True)


# In[42]:


df['Active_Loan'].fillna(df['Active_Loan'].median(),inplace=True)


# In[43]:


df['House_Own'].fillna(df['House_Own'].median(),inplace=True)


# In[44]:


df['Child_Count'].fillna(df['Child_Count'].median(),inplace=True)


# In[45]:


df['Credit_Amount'].fillna(df['Credit_Amount'].median(),inplace=True)


# In[46]:


df['Loan_Annuity'].fillna(df['Loan_Annuity'].median(),inplace=True)


# In[47]:


df['Population_Region_Relative'].fillna(df['Population_Region_Relative'].median(),inplace=True)


# In[48]:


df['Age_Days'].fillna(df['Age_Days'].median(),inplace=True)


# In[49]:


df['Employed_Days'].fillna(df['Employed_Days'].median(),inplace=True)


# In[50]:


df['Registration_Days'].fillna(df['Registration_Days'].median(),inplace=True)


# In[51]:


df['ID_Days'].fillna(df['ID_Days'].median(),inplace=True)


# In[52]:


df['Cleint_City_Rating'].fillna(df['Cleint_City_Rating'].median(),inplace=True)


# In[53]:


df['Client_Family_Members'].fillna(df['Client_Family_Members'].median(),inplace=True)


# In[54]:


df['Application_Process_Day'].fillna(df['Application_Process_Day'].median(),inplace=True)


# In[55]:


df['Application_Process_Hour'].fillna(df['Application_Process_Hour'].median(),inplace=True)


# In[56]:


df['Score_Source_3'].fillna(df['Score_Source_3'].median(),inplace=True)


# In[57]:


df['Car_Owned'].fillna(df['Car_Owned'].median(),inplace=True)


# In[58]:


df['Own_House_Age'].fillna(df['Own_House_Age'].median(),inplace=True)


# In[59]:


df['Score_Source_1'].fillna(df['Score_Source_1'].median(),inplace=True)


# In[60]:


df['Score_Source_2'].fillna(df['Score_Source_2'].median(),inplace=True)


# In[61]:


df['Social_Circle_Default'].fillna(df['Social_Circle_Default'].median(),inplace=True)


# In[62]:


df['Phone_Change'].fillna(df['Phone_Change'].median(),inplace=True)


# In[63]:


df['Credit_Bureau'].fillna(df['Credit_Bureau'].median(),inplace=True)


# In[64]:


a=df.isnull().sum()
prec=(a/(len(df)))*100
prec=pd.DataFrame(prec,columns=['Age of missing data'])
prec


# In[65]:


prec=df.apply(lambda x:x/x.sum()*100)


# In[66]:


print(prec)

#prepare our dataset
# In[67]:


df.info()


# In[68]:


df.isnull().sum()


# In[69]:


df.head()


# In[70]:


df.tail()


# # SVM

# In[71]:


#x=df.drop(['Client_Education','Accompany_Client','Client_Income_Type','Client_Marital_Status','Client_Gender'
#           ,'Loan_Contract_Type','Client_Housing_Type','Client_Occupation','Client_Permanent_Match_Tag'
#           ,'Client_Contact_Work_Tag','Type_Organization'],axis=1)

#x.head()


# In[72]:


x=df.drop(['Default'],axis=1)


# In[73]:


y=df[['Default']]


# In[74]:


from sklearn.model_selection import train_test_split


# In[75]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.5, random_state=45)


# In[76]:


x_train.shape


# In[77]:


x_test.shape


# In[78]:


y_train.shape


# In[79]:


y_test.shape


# In[80]:


from sklearn.svm import SVC


# In[81]:


model=SVC(kernel='linear',C=1.0,gamma='auto')


# In[82]:


model.fit(x_train,y_train)


# In[83]:


y_pred=model.predict(x_test)
y_pred


# In[ ]:





# In[84]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[85]:


accuracy_score(y_test,y_pred)*100


# In[86]:


cf=confusion_matrix(y_test,y_pred)
cf


# In[87]:


print(classification_report(y_test,y_pred))


# In[88]:


plt.figure(figsize=(10,5))
plt.title('confusion_matrix_SVM',fontsize=20)
sns.heatmap(cf,annot=True,cmap='Blues',fmt='g')
plt.show()


# In[ ]:





# # Navie Bayes method

# In[89]:


x=df.drop(['Default'],axis=1)

x.head()


# In[90]:


y=df[['Default']]

y.head()


# In[91]:


from sklearn.model_selection import train_test_split


# In[92]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[93]:


x_train.shape


# In[94]:


x_test.shape


# In[95]:


y_train.shape


# In[96]:


y_test.shape


# In[97]:


from sklearn.naive_bayes import GaussianNB


# In[98]:


model=GaussianNB()


# In[99]:


model.fit(x_train,y_train)


# In[100]:


y_pred=model.predict(x_test)
y_pred


# In[101]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[102]:


accuracy_score(y_test,y_pred)*100


# In[103]:


from sklearn.linear_model import LogisticRegression


# In[104]:


lr=LogisticRegression()


# In[105]:


lr.fit(x_train,y_train)


# In[106]:


lr_pred=lr.predict(x_test)
lr_pred


# In[107]:


accuracy_score(y_test,lr_pred)*100


# In[108]:


from sklearn.neighbors import KNeighborsClassifier


# In[109]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[110]:


knn.fit(x_train,y_train)


# In[111]:


knn_pred=knn.predict(x_test)
knn_pred


# In[112]:


accuracy_score(y_test,knn_pred)*100


# In[113]:


cf=confusion_matrix(y_test,y_pred)
cf


# In[114]:


plt.figure(figsize=(15,5))
plt.title('confussion_matrix_NB',fontsize=20)
sns.heatmap(cf,annot=True,fmt='g',cmap='Blues')
plt.show()


# In[115]:


print(classification_report(y_test,y_pred))


# In[ ]:





# # Decision Tree

# In[116]:


x=df.drop(['Default'],axis=1)
y=df[['Default']]


# In[117]:


from sklearn.model_selection import train_test_split


# In[118]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[119]:


x_train.shape


# In[120]:


x_test.shape


# In[121]:


y_train.shape


# In[122]:


y_test.shape


# In[123]:


from sklearn.tree import DecisionTreeClassifier


# In[124]:


model=DecisionTreeClassifier(criterion='entropy',max_depth=9)


# In[125]:


model.fit(x_train,y_train)


# In[126]:


y_pred=model.predict(x_test)


# In[127]:


model.score(x_train,y_train)*100   


# In[128]:


model.score(x_test,y_test)*100


# In[129]:


from sklearn.metrics import confusion_matrix,classification_report


# In[130]:


cf=confusion_matrix(y_test,y_pred)


# In[131]:


#plotting the confusion matrix
plt.figure(figsize=(10,5))
plt.title('confusion_matrix_flight',fontsize=20)
sns.heatmap(cf,annot=True,fmt='g',cmap='Blues')
plt.show()


# In[132]:


((33414+89)/(33414+89+149+2905))*100


# In[133]:


print(classification_report(y_test,y_pred))


# In[134]:


from sklearn.tree import plot_tree


# In[135]:


f=list(x_train)
c=['No','Yes']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(15,8),dpi=300)
plot_tree(model,feature_names=f,class_names=c,filled=True)
plt.title('Decision_Tree_classificer_Entropy',fontsize=20)
plt.show()


# In[ ]:




