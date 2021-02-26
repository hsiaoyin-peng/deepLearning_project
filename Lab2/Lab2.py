#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start_time = time.time()


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.jpg'):
            break
        print(os.path.join(dirname, filename))


# In[3]:


sample_sub = pd.read_csv('D:\herbarium-2020-fgvc7/sample_submission.csv')
print(sample_sub)


# In[4]:


import json, codecs
with codecs.open("D:\herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    train_meta = json.load(f)
    
with codecs.open("D:\herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    test_meta = json.load(f)


# In[5]:


print(train_meta.keys())


# In[6]:


train_df = pd.DataFrame(train_meta['annotations'])
print(train_df)


# In[7]:


train_cat = pd.DataFrame(train_meta['categories'])
train_cat.columns = ['family', 'genus', 'category_id', 'category_name']
print(train_cat)


# In[8]:


train_img = pd.DataFrame(train_meta['images'])
train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']
print(train_img)


# In[9]:


train_reg = pd.DataFrame(train_meta['regions'])
train_reg.columns = ['region_id', 'region_name']
print(train_reg)


# In[10]:


train_df = train_df.merge(train_cat, on='category_id', how='outer')
train_df = train_df.merge(train_img, on='image_id', how='outer')
train_df = train_df.merge(train_reg, on='region_id', how='outer')


# In[11]:


print(train_df.info())
print(train_df)


# In[12]:


na = train_df.file_name.isna()
keep = [x for x in range(train_df.shape[0]) if not na[x]]
train_df = train_df.iloc[keep]


# In[13]:


dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32', 'object']
for n, col in enumerate(train_df.columns):
    train_df[col] = train_df[col].astype(dtypes[n])
print(train_df.info())
print(train_df)


# In[14]:


test_df = pd.DataFrame(test_meta['images'])
test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']
print(test_df.info())
test=test_df
print(test_df)


# In[15]:


print("Total Unique Values for each columns:")
print("{0:10s} \t {1:10d}".format('train_df', len(train_df)))
for col in train_df.columns:
    print("{0:10s} \t {1:10d}".format(col, len(train_df[col].unique())))


# In[16]:


family = train_df[['family', 'genus', 'category_name']].groupby(['family', 'genus']).count()
print(family.describe())


# In[17]:


train_df.pop('id')
train_df.pop('image_id')
train_df.pop('region_id')
train_df.pop('family')
train_df.pop('genus')
train_df.pop('category_name')
train_df.pop('height')
train_df.pop('license')
train_df.pop('width')
train_df.pop('region_name')
train = train_df.sort_values('category_id')[:6000]


# In[23]:


# Import Keras libraries and packages
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten,BatchNormalization,Dropout
from keras.layers import Dense # Fully Connected Networks

# initializing CNN
model = Sequential()  
model.add(Conv2D(3, (3, 3), input_shape = (120, 120, 3), activation = 'relu', padding='same', kernel_initializer='random_normal'))
model.add(Conv2D(3, (5, 5), input_shape = (120, 120, 3), activation = 'relu', padding='same', kernel_initializer='random_normal'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(3,3)))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# In[24]:


# Second convolutional layer
model.add(Conv2D(16, (5, 5), strides=(5,5)))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# In[25]:


model.add(Flatten())


# In[26]:


model.add(Dense(300, activation = 'softmax'))
model.add(Dense(32093, activation = 'softmax'))
print(model.summary())


# In[22]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[106]:


from sklearn.model_selection import train_test_split as tts
train, verif = tts(train, test_size=0.2, shuffle=True, random_state=17)


# In[107]:


from keras.preprocessing.image import ImageDataGenerator

shape = (120, 120, 3)
epochs = 2
batch_size = 32

train_datagen = ImageDataGenerator(featurewise_center=False,
                                   featurewise_std_normalization=False,
                                   rotation_range=180,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2)


# In[108]:


test_datagen = ImageDataGenerator(featurewise_center=False,
                                  featurewise_std_normalization=False)


# In[109]:


training_set = train_datagen.flow_from_dataframe(dataframe=train,
                                                 directory='D:\herbarium-2020-fgvc7/nybg2020/train/',
                                                 x_col="file_name",
                                                 y_col=["category_id"],
                                                 target_size=(120, 120),
                                                 batch_size=batch_size,
                                                 class_mode='multi_output')


# In[110]:


valid_set = train_datagen.flow_from_dataframe(dataframe=verif,
                                              directory='D:\herbarium-2020-fgvc7/nybg2020/train/',
                                              x_col="file_name",
                                              y_col=["category_id"],
                                              target_size=(120, 120),
                                              batch_size=batch_size,
                                              class_mode='multi_output')


# In[111]:


model.fit_generator(training_set,
                    validation_data=valid_set,
                    epochs=epochs,
                    steps_per_epoch=len(train),
                    validation_steps=len(verif),
                    verbose=1,
                    workers=8)


# In[112]:


model.save('test_model.h5')


# In[113]:


batch_size = 32
test_datagen = ImageDataGenerator(featurewise_center=False,
                                  featurewise_std_normalization=False)

generator = test_datagen.flow_from_dataframe(dataframe = test,
                                             directory = 'D:\herbarium-2020-fgvc7/nybg2020/test/',
                                             x_col = 'file_name',
                                             target_size=(120, 120),
                                             batch_size=batch_size,
                                             class_mode=None,  # only data, no labels
                                             shuffle=False)

category = model.predict_generator(generator, verbose=1)


# In[114]:


sub = pd.DataFrame()
sub['Id'] = test.image_id
sub['Id'] = sub['Id'].astype('int32')
sub['Predicted'] = np.concatenate([np.argmax(category, axis=1), 23718*np.ones((len(test.image_id)-len(category)))], axis=0)
sub['Predicted'] = sub['Predicted'].astype('int32')
print(sub)
sub.to_csv('category_submission.csv', index=False)

