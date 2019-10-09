# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
h2o.init()
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,Normalizer
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from pylab import rcParams
rcParams['figure.figsize']=15,10

student=pd.read_csv('studentDropIndia_20161215.csv', sep=',')
student.isnull().any()

student=student.fillna(0)

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(student.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#plt.savefig('corr_matrix.png', format='png')

# Create correlation matrix
#corr_matrix = df.corr().abs()
#
## Select upper triangle of correlation matrix
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#
## Find features with correlation greater than 0.95
#to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#
## Drop features 
#df.drop(df.columns[to_drop], axis=1)

labels = ['continue', 'drop']
sizes = [student['continue_drop'].value_counts()[0],
         student['continue_drop'].value_counts()[1]
        ]
fig1, ax1 = plt.subplots()

##Plot as a pie for comparison
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.title('Continue vs Dropout Pie Chart', fontsize=20)
plt.show()
plt.savefig('pie_relation.jpg', format='jpg')

##Plot as a bar for comparison
#maxim = max(student['continue_drop'].value_counts()[0], student['continue_drop'].value_counts()[1])
#yax = (0,1)
#contt = student['continue_drop'].value_counts()[0]/maxim
#dropp = student['continue_drop'].value_counts()[1]/maxim
#student.plot.bar(contt, dropp) 
#plt.show()
#plt.figure()
#plt.savefig('hist_relation.png', format='png')

#We make a custom list sam eas the size of the number of columns(features) in the given data
predictors=list(range(0,15))
print(student.shape)

cols_to_transform = [ 'continue_drop','gender','caste','guardian','internet' ]
student = pd.get_dummies( student,columns = cols_to_transform )
student.head()

student = student.drop('student_id', 1)

# Copy the original dataset
scaled_features = student.copy()

# Extract column names to be standardized
col_names = ['mathematics_marks','english_marks','science_marks',
             'science_teacher','languages_teacher','school_id',
             'total_students','total_toilets','establishment_year'#,
             #'gender_F','gender_M','caste_BC','caste_OC','caste_SC',
             #'caste_ST','guardian_father','guardian_mixed','guardian_mother',
            # 'guardian_other','internet_False','internet_True'
            ]

# Standardize the columns and re-assingn to original dataframe
features = scaled_features[col_names]
scaler = RobustScaler().fit_transform(features.values)
features = pd.DataFrame(scaler, index=student.index, columns=col_names)
scaled_features [col_names] = features
scaled_features.head()

#student = student.astype(object)

train=scaled_features.loc[scaled_features['continue_drop_continue'] == 1]
test=scaled_features.loc[scaled_features['continue_drop_drop'] == 1]

h2o.init(nthreads=-1, enable_assertions = False)

train.hex=h2o.H2OFrame(train)
test.hex=h2o.H2OFrame(test)

model=H2OAutoEncoderEstimator(activation="Tanh",
                              hidden=[120],
                              ignore_const_cols=False,
                              epochs=100
                             )

model.train(x=predictors,training_frame=train.hex)
model._model_json['output']

test_rec_error=model.anomaly(test.hex)
train_rec_error=model.anomaly(train.hex)

test_rec_error_df=test_rec_error.as_data_frame()
train_rec_error_df=train_rec_error.as_data_frame()
final = pd.concat([train_rec_error_df, test_rec_error_df])

boxplotEdges=final.quantile(.75)
iqr = np.subtract(*np.percentile(final, [75, 25]))
top_whisker=boxplotEdges[0]+(1.5*iqr)
top_whisker

train_rec_error_df['id']=train_rec_error_df.index
test_rec_error_df['id']=test_rec_error_df.index + 18200 #Count of train data

plt.scatter(train_rec_error_df['id'],train_rec_error_df['Reconstruction.MSE'],label='Continued Students',s=1)
plt.axvline(x=18200,linewidth=1)
plt.scatter(test_rec_error_df['id'],test_rec_error_df['Reconstruction.MSE'],label='Dropped Students',s=1)
plt.axhline(y=top_whisker,linewidth=1, color='r')
plt.legend()

h2o.cluster().shutdown()

