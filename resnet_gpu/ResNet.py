#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
import numpy as np
import os
import random
import shutil

from glob import glob




def load_data_files(base_dir):
    RAW_DATASET = os.path.join(base_dir,"flower_photos")

    sub_dir = map(lambda d: os.path.basename(d.rstrip("/")), glob(os.path.join(RAW_DATASET,'*/')))

    data_dic = {}
    for class_name  in sub_dir:
        imgs = glob(os.path.join(RAW_DATASET,class_name,"*.jpg"))

        data_dic[class_name] = imgs
        print("Class: {}".format(class_name))
        print("Number of images: {} \n".format(len(imgs)))

    return data_dic



def train_validation_split(base_dir, data_dic, split_ratio=0.2):
    FLOWER_DATASET = os.path.join(base_dir,"flower_dataset")

    if not os.path.exists(FLOWER_DATASET):
        os.makedirs(FLOWER_DATASET)

    for class_name, imgs in data_dic.items():
        idx_split = int(len(imgs) * split_ratio)
        random.shuffle(imgs)
        validation = imgs[:idx_split]
        train = imgs[idx_split:]

        copy_files_to_directory(train, os.path.join(FLOWER_DATASET,"train",class_name))
        copy_files_to_directory(validation, os.path.join(FLOWER_DATASET,"validation",class_name))


def copy_files_to_directory(files, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created directory: {}".format(directory))

    for f in files:
        shutil.copy(f, directory)
    print("Copied {} files.\n".format(len(files)))        
    
    
    
def residual_block(input_tensor, filters, stage, reg=0.0, use_shortcuts=True):

        bn_name = 'bn' + str(stage)
        conv_name = 'conv' + str(stage)
        relu_name = 'relu' + str(stage)
        merge_name = 'merge' + str(stage)

        # 1x1 conv
        # batchnorm-relu-conv
        # from input_filters to bottleneck_filters
        if stage>1: # first activation is just after conv1
            x = tf.keras.layers.BatchNormalization(name=bn_name+'a')(input_tensor)
            x = tf.keras.layers.Activation('relu', name=relu_name+'a')(x)
        else:
            x = input_tensor

        x = tf.keras.layers.Convolution2D(
                filters[0], (1,1),
                kernel_regularizer = tf.keras.regularizers.l2(reg),
                use_bias=False,
                name=conv_name+'a'
            )(x)

        # 3x3 conv
        # batchnorm-relu-conv
        # from bottleneck_filters to bottleneck_filters
        x = tf.keras.layers.BatchNormalization(name=bn_name+'b')(x)
        x = tf.keras.layers.Activation('relu', name=relu_name+'b')(x)
        x = tf.keras.layers.Convolution2D(
                filters[1], (3,3),
                padding='same',
                kernel_regularizer = tf.keras.regularizers.l2(reg),
                use_bias = False,
                name=conv_name+'b'
            )(x)

        # 1x1 conv
        # batchnorm-relu-conv
        # from bottleneck_filters  to input_filters
        x = tf.keras.layers.BatchNormalization(name=bn_name+'c')(x)
        x = tf.keras.layers.Activation('relu', name=relu_name+'c')(x)
        x = tf.keras.layers.Convolution2D(
                filters[2], (1,1),
                kernel_regularizer = tf.keras.regularizers.l2(reg),
                name=conv_name+'c'
            )(x)

        # merge output with input layer (residual connection)
        if use_shortcuts:
            x = tf.keras.layers.add([x, input_tensor], name=merge_name)

        return x    
    
    
        
class Res_net:

    def __init__(self):

        self.models = tf.keras.models
        self.layers = tf.keras.layers
        self.initializers = tf.keras.initializers
        self.regularizers = tf.keras.regularizers
        self.losses = tf.keras.losses
        self.optimizers = tf.keras.optimizers 
        self.metrics = tf.keras.metrics
        self.preprocessing_image = tf.keras.preprocessing.image
        self.callbacks = tf.keras.callbacks




   


    def ResNetPreAct(self, input_shape=(150,150,3), nb_classes=6, num_stages=5,
                    use_final_conv=True, reg=0.0):


        # Input
        img_input = self.layers.Input(shape=input_shape, name='input_intel_1')

        #### Input stream ####
        # conv-BN-relu-(pool)
        x = self.layers.Convolution2D(
                256, (3,3), strides=(2, 2),
                padding='same',
                kernel_regularizer=self.regularizers.l2(reg),
                use_bias=False,
                name='conv0'
            )(img_input)
        x = self.layers.BatchNormalization(name='bn0')(x)
        x = self.layers.Activation('relu', name='relu0')(x)
        x = self.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool0')(x)

        #### Residual Blocks ####
        # 1x1 conv: batchnorm-relu-conv
        # 3x3 conv: batchnorm-relu-conv
        # 1x1 conv: batchnorm-relu-conv
        for stage in range(1,num_stages+1):         
            x = residual_block(x, [32,64,256], stage=stage, reg=reg)    ###여기부터 과제임
        
        x = self.layers.MaxPooling2D((3, 3), strides=(3, 3), padding='same', name='pool1')(x)
            
        #### Output stream ####
        # BN-relu-(conv)-avgPool-softmax
        x = self.layers.BatchNormalization(name='bnF')(x)
        x = self.layers.Activation('relu', name='reluF')(x)

        # Optional final conv layer
        if use_final_conv:
            x = self.layers.Convolution2D(
                    512, (5,5),
                    padding='valid',
                    kernel_regularizer=self.regularizers.l2(reg),
                    name='convF'
                )(x)
            
            x = self.layers.BatchNormalization(name='bn0_added')(x)
            x = self.layers.Activation('relu', name='relu0_added')(x)
        
        
        #pool_size = input_shape[0] / 2
        x = self.layers.GlobalAveragePooling2D(name='global_avg_pool_intel')(x)

        x = self.layers.Dropout(name='dropout', rate=0.2)(x)
        x = self.layers.Dense(nb_classes, activation='softmax', name='fc10')(x)

        return self.models.Model(img_input, x, name='rnpa')


    def compile_model(self,model):
        loss = self.losses.categorical_crossentropy


        optimizer = self.optimizers.Adam(learning_rate=0.0001, amsgrad=True)

        # metrics
        metric = ['acc','categorical_accuracy']


        # compile model with loss, optimizer, and evaluation metrics
        model.compile(optimizer, loss, metric)

        return model

    def validation_image_processing(self,x):
        test_datagen = self.preprocessing_image.ImageDataGenerator(rescale=1./255)
        validation_generator = test_datagen.flow_from_directory(
            x,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            seed=1997)
        
        return validation_generator

    def train_image_processing(self,x):
	
        train_datagen = self.preprocessing_image.ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.1,rotation_range=15,horizontal_flip=True,vertical_flip=True)
        train_generator = train_datagen.flow_from_directory(
            x,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            seed=1997)
            
        return train_generator



    def starting(self,train_generator,validation_generator,model,base_dir):
        
        if os.path.isdir('checkpoints') != True:
            os.mkdir('checkpoints')
        
        checkpoint_filepath = base_dir + '/checkpoints'

        model_checkpoint_callback = self.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='categorical_accuracy',
            mode='max',
            save_best_only=True)

        checkpoints = self.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + '/best.h5', monitor='categorical_accuracy', mode='max',save_weights_only=False,save_best_only=True)

        tensorboard_callbacks = self.callbacks.TensorBoard(log_dir='logs', histogram_freq=2, write_graph=True, update_freq='epoch', profile_batch=2, embeddings_freq=0)
        early_stoping = self.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=5, verbose=1, mode='max')
            
            
        model.fit(
        train_generator,
        steps_per_epoch = len(train_generator),
        epochs=100,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoints, tensorboard_callbacks, early_stoping])
        
        return True